import pyomo.environ as pyo
from pyomo.environ import value, minimize
import pandas as pd
import numpy as np

# Importamos las dependencias correspondientes al modelo determinista
from Pisco_Det_Forecasting import PiscoModel_Operativo 
from Integrar_Forecasting import cargar_demanda_determinista, cargar_escenario_101
from solver_utils import GurobiConfigurationError, create_gurobi_solver

class PiscoRollingModel:
    def __init__(self, data_path, pronostico_path, semilla_realidad=1001):
        self.pronostico_path = pronostico_path
        self.semilla_realidad = semilla_realidad
        self.modelo_base = PiscoModel_Operativo()
        self.modelo_base.ReadExcelFile(data_path)
        self.instance = None
        self.dia_calendario = 0
        
        # Estructuras para guardar ambas fases, igual que en el modelo estocástico
        self.historial_costos = []
        self.historial_costos_in_sample = []

        # Esto crea el atributo 'Demanda_Determinista' en el objeto 'modelo_base'
        self.modelo_base.Cargar_Pronostico_Determinista(pronostico_path, self.dia_calendario)

        df_pronosticos_full = pd.read_excel(pronostico_path, engine='openpyxl')
        
        # Diccionario para búsqueda rápida de pronósticos F_t
        self.pronosticos_dict = {}
        for _, row in df_pronosticos_full.iterrows():
            mezcla = row['n']
            fecha_hoy = int(row['Fecha de Hoy'])
            dia_entrega = int(fecha_hoy + row['t'])
            self.pronosticos_dict[(mezcla, fecha_hoy, dia_entrega)] = max(0, row['F_t'])
        
    def Construir_Modelo_Nerviosismo(self, RP=30, C_nerv_val=5.0):

        self.instance = self.modelo_base.Problema()
        inst = self.instance 
        
        inst.RP = pyo.Param(initialize=RP, mutable=True)
        inst.C_nerv = pyo.Param(initialize=C_nerv_val, mutable=True)
        inst.q_bar = pyo.Param(inst.A, inst.Tp, initialize=0.0, mutable=True)
        inst.f = pyo.Var(inst.A, inst.Tp, within=pyo.NonNegativeReals, initialize=0.0)
        
        def lin_nerv_1_rule(model, a, t):
            if t > len(model.Tp) - pyo.value(model.RP):
                return pyo.Constraint.Skip
            return model.f[a, t] >= model.q[a, t] - model.q_bar[a, t]
        inst.const_nerv_1 = pyo.Constraint(inst.A, inst.Tp, rule=lin_nerv_1_rule)

        def lin_nerv_2_rule(model, a, t):
            if t > len(model.Tp) - pyo.value(model.RP):
                return pyo.Constraint.Skip
            return model.f[a, t] >= model.q_bar[a, t] - model.q[a, t]
        inst.const_nerv_2 = pyo.Constraint(inst.A, inst.Tp, rule=lin_nerv_2_rule)

        inst.FO.deactivate()

        def obj_rolling_rule(model):
            costo_original = model.FO.expr
            costo_nerviosismo = sum(model.C_nerv * model.f[a, t] for a in model.A
                for t in model.Tp if t <= len(model.Tp) - pyo.value(model.RP))
            return costo_original + costo_nerviosismo

        inst.FO_Rolling = pyo.Objective(rule=obj_rolling_rule, sense=minimize)

        return inst

    def _valor_variable(self, variable, default=0.0):
        valor = pyo.value(variable, exception=False)
        if valor is None:
            return default
        if abs(valor) < 1e-6:
            return 0.0
        return float(valor)

    def _periodos_rp(self, inst):
        RP = int(pyo.value(inst.RP))
        return [t for t in sorted(inst.Tp) if t <= RP]

    def _obtener_nombre_mezcla(self, n):
        return self.modelo_base.Mezclas.iloc[n-1]['n']

    def _solucion_exitosa(self, results):
        return (
            results.solver.status == pyo.SolverStatus.ok
            and results.solver.termination_condition in [
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.maxTimeLimit,
            ]
        )

    def Guardar_Plan_Q(self, inst):
        return {(a, t): self._valor_variable(inst.q[a, t]) for a in inst.A for t in inst.Tp}

    def Congelar_MPS_RP(self, inst):
        L_lote = pyo.value(inst.L_lote) if hasattr(inst, 'L_lote') else None
        for n in inst.N:
            for t in self._periodos_rp(inst):
                inst.y[n, t].fix(self._valor_variable(inst.y[n, t]))

        for a in inst.A:
            for t in self._periodos_rp(inst):
                q_val = self._valor_variable(inst.q[a, t])
                if hasattr(inst, 'l'):
                    l_val = round(self._valor_variable(inst.l[a, t]))
                    inst.l[a, t].fix(l_val)
                    if L_lote is not None:
                        q_val = L_lote * l_val
                inst.q[a, t].fix(q_val)

    def Liberar_MPS_RP(self, inst):
        for n in inst.N:
            for t in self._periodos_rp(inst):
                inst.y[n, t].unfix()

        for a in inst.A:
            for t in self._periodos_rp(inst):
                inst.q[a, t].unfix()
                if hasattr(inst, 'l'):
                    inst.l[a, t].unfix()

    def Inyectar_Demanda_Real_OutOfSample(self, inst, pronostico_path, semilla_101):
        # CAMBIO CLAVE: Inyectamos el Escenario 101 Sintético como OOS
        print(f"Inyectando Escenario 101 Sintético (Realidad) para evaluar Día: {self.dia_calendario}")
        
        demanda_realidad_101 = cargar_escenario_101(
            ruta_csv=pronostico_path, 
            dia_actual=self.dia_calendario,
            semilla_realidad=semilla_101
        )
        
        for n in inst.N:
            n_name = self._obtener_nombre_mezcla(n)
            for t in inst.Tp:
                dia_real = self.dia_calendario + t
                inst.D_nt[n, t] = demanda_realidad_101.get((n_name, dia_real), 0.0)

    def Calcular_Costos_Reales_RP(self, inst):
        periodos = self._periodos_rp(inst)
        cv = pyo.value(inst.Cv_cost)
        ci_minus = pyo.value(inst.Ci_minus_cost)
        c_nerv = pyo.value(inst.C_nerv)

        costo_i = sum(cv * self._valor_variable(inst.v[a, t]) for a in inst.A for t in periodos)
        costo_ii = sum(
            pyo.value(inst.Cwip_n[n, p]) * self._valor_variable(inst.z[n, p, t, u])
            for (n, p) in inst.Pn
            for t in periodos
            for u in inst.U
            if u < pyo.value(inst.Tau_np[n, p])
        )
        costo_iii = sum(
            pyo.value(inst.Cwip_n[n, p]) * self._valor_variable(inst.k[n, p, t])
            for (n, p) in inst.Pn
            for t in periodos
        )
        costo_iv = sum(ci_minus * self._valor_variable(inst.i_minus[n, t]) for n in inst.N for t in periodos)
        costo_nerv = sum(c_nerv * self._valor_variable(inst.f[a, t]) for a in inst.A for t in periodos)
        total = costo_i + costo_ii + costo_iii + costo_iv + costo_nerv

        return {
            'Costo_Real_Almacenamiento_I_RP': costo_i,
            'Costo_Real_Maduracion_II_RP': costo_ii,
            'Costo_Real_Nerviosismo_RP': costo_nerv,
            'Costo_Real_Stock_Terminado_III_RP': costo_iii,
            'Costo_Real_Backlog_IV_RP': costo_iv,
            'Costo_Real_Total_RP': total,
        }

    def Exportar_Historial_Costos(self, nombre_archivo="Historial_Costos_RH_Determinista_OOS.xlsx"):
        if not self.historial_costos:
            print("No hay historial OOS para exportar.")
            return

        df_historial = pd.DataFrame(self.historial_costos)
        print(f"\nExportando historial OOS a '{nombre_archivo}'...")
        df_historial.to_excel(nombre_archivo, index=False, engine='openpyxl')
        print("Exportación de historial OOS completada.")

    def Exportar_Historial_Costos_InSample(self, nombre_archivo="Historial_Costos_RH_Determinista_InSample.xlsx"):
        if not self.historial_costos_in_sample:
            print("No hay historial in-sample para exportar.")
            return

        df_historial = pd.DataFrame(self.historial_costos_in_sample)
        print(f"\nExportando historial in-sample a '{nombre_archivo}'...")
        df_historial.to_excel(nombre_archivo, index=False, engine='openpyxl')
        print("Exportación de historial in-sample completada.")

    def Actualizar_Estado_Inicial(self, inst, q_plan_referencia=None):
        RP = int(pyo.value(inst.RP))
        max_t = max(inst.Tp)
        
        for a in inst.A:
            inst.I_a[a] = pyo.value(inst.v[a, RP])
            inst.W_a[a] = pyo.value(inst.q[a, RP])          
            
        for n in inst.N:
            inst.I_minus_n[n] = pyo.value(inst.i_minus[n, RP])
            
        for (n, p) in inst.Pn:
            tau_val = int(pyo.value(inst.Tau_np[n, p]))
            for u in inst.U:
                if u < tau_val:
                    inst.gamma[n, p, u] = pyo.value(inst.z[n, p, RP, u])
                    
        for (g_name, p_id) in inst.Set_Grupos_P:
            lista_mezclas_ids = self.modelo_base.Group_Mapping.get((g_name, p_id), [])
            suma_k = sum(pyo.value(inst.k[n_id, p_id, RP]) for n_id in lista_mezclas_ids)
            inst.I_Gp[g_name, p_id] = suma_k

        for a in inst.A:
            for t in inst.Tp:
                if t + RP <= max_t:
                    if q_plan_referencia is not None:
                        inst.q_bar[a, t] = q_plan_referencia.get((a, t + RP), 0.0)
                    else:
                        inst.q_bar[a, t] = pyo.value(inst.q[a, t + RP])
                else:
                    inst.q_bar[a, t] = 0.0
                    
        self.dia_calendario += RP
        
        print(f"Actualizando pronóstico para el día calendario: {self.dia_calendario}")
        for n in inst.N:
            n_name = self.modelo_base.Mezclas.iloc[n-1]['n']
            for t in inst.Tp:
                dia_real = self.dia_calendario + t
                valor_forecast = self.pronosticos_dict.get((n_name, self.dia_calendario, dia_real), 0.0)
                inst.D_nt[n, t] = valor_forecast

    def Ejecutar_Ciclo_Rolling_Horizon(self, iteraciones=4):

        inst = self.Construir_Modelo_Nerviosismo(RP=30, C_nerv_val=5.0)
        try:
            solver = create_gurobi_solver({"TimeLimit": 900, "MIPGap": 0.05})
        except GurobiConfigurationError as exc:
            print(f"\nError de configuración de Gurobi:\n{exc}\n")
            return

        for iteracion in range(iteraciones):
            if iteracion == 0:
                inst.C_nerv.value = 0.0 
            else:
                inst.C_nerv.value = 5.0 
                
            # --- FASE 1: PLANIFICACIÓN A CIEGAS (IN-SAMPLE) ---
            results = solver.solve(inst, tee=False)
            
            if self._solucion_exitosa(results):
                
                RP_val = pyo.value(inst.RP)
                c_nerv_val = pyo.value(inst.C_nerv)
                costo_nerv_iter = sum(c_nerv_val * pyo.value(inst.f[a, t]) 
                                      for a in inst.A for t in inst.Tp 
                                      if t <= len(inst.Tp) - RP_val)

                print(f'\n' + '='*50)
                print(f" ITERACIÓN ROLLING HORIZON DET:   {iteracion}")
                print('='*50)
                print(' 1️⃣ COSTOS OPERATIVOS (Primera Etapa / Fijos):')
                print(f'  - Costo I (Almacenamiento):         $ {pyo.value(inst.coste_I):,.2f}')
                print(f'  - Costo II (Inventario Maduración): $ {pyo.value(inst.coste_II):,.2f}')
                print(f'  - Costo de Nerviosismo (Cambios):   $ {costo_nerv_iter:,.2f}')
                
                total_1ra = pyo.value(inst.coste_I) + pyo.value(inst.coste_II) + costo_nerv_iter
                print(f" Subtotal Operativo:                  $ {total_1ra:,.2f}")
                print('-'*50)
                print(' 2️⃣ COSTOS DE STOCK Y PENALIZACIÓN (Proyectados IS):')
                print(f'  - Costo III (Stock Terminado):      $ {pyo.value(inst.coste_III):,.2f}')
                print(f'  - Costo IV (Multas Backlog):        $ {pyo.value(inst.coste_IV):,.2f}')
                total_2da = pyo.value(inst.coste_III) + pyo.value(inst.coste_IV)
                print(f" Subtotal Penalizaciones y Stock:     $ {total_2da:,.2f}")
                print('-'*50)
                print(f" COSTO TOTAL FO (Rolling In-Sample):  $ {pyo.value(inst.FO_Rolling):,.2f}")
                print('='*50 + '\n')
            else:
                print(f"Error: Solver no convergió en Fase IS. Estado: {results.solver.termination_condition}")
                break

            costo_planificacion_total = pyo.value(inst.FO_Rolling)
            q_plan_referencia = self.Guardar_Plan_Q(inst)

            # --- EXPORTAR SOLUCIÓN IN-SAMPLE ---
            nombre_excel_is = f"Resultados_RH_Det_IS_Iteracion_{iteracion}.xlsx"
            print(f"Exportando resultados in-sample de la iteración a: {nombre_excel_is}")
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel_is)

            fila_in_sample = {
                "Iteracion": iteracion,
                "Dia_Inicio": self.dia_calendario,
                "Dia_Fin_RP": self.dia_calendario + int(pyo.value(inst.RP)),
                "Escenario_InSample": "Pronostico_Promedio_Ft",
                "Costo_I": float(pyo.value(inst.coste_I)),
                "Costo_II": float(pyo.value(inst.coste_II)),
                "Costo_Nerviosismo": float(costo_nerv_iter),
                "Costo_III": float(pyo.value(inst.coste_III)),
                "Costo_IV": float(pyo.value(inst.coste_IV)),
                "Costo_Total_InSample": float(pyo.value(inst.FO_Rolling)),
                "Escenario_Evaluacion": "InSample_Det",
            }
            self.historial_costos_in_sample.append(fila_in_sample)


            # --- FASE 2: EVALUACIÓN OUT-OF-SAMPLE (CHOQUE CON LA REALIDAD) ---
            self.Congelar_MPS_RP(inst)
            self.Inyectar_Demanda_Real_OutOfSample(
                inst, 
                pronostico_path=self.pronostico_path, 
                semilla_101=self.semilla_realidad
            )
            
            results_real = solver.solve(inst, tee=False)

            if not self._solucion_exitosa(results_real):
                print(f"Error: evaluación real OOS no convergió. Estado: {results_real.solver.termination_condition}")
                self.Liberar_MPS_RP(inst)
                break

            costos_reales = self.Calcular_Costos_Reales_RP(inst)
            datos_iteracion_oos = {
                'Iteracion': iteracion,
                'Modelo': 'Determinista',
                'Dia_Inicio': self.dia_calendario,
                'Dia_Fin_RP': self.dia_calendario + int(pyo.value(inst.RP)),
                'Escenario_InSample': 'Pronostico_Promedio_Ft',
                'Escenario_Evaluacion_OOS': 'Escenario_101_Sintetico',
                'Costo_Planificacion_Total_HL': costo_planificacion_total,
            }
            datos_iteracion_oos.update(costos_reales)
            self.historial_costos.append(datos_iteracion_oos)

            print('\n' + '='*50)
            print(f" EVALUACIÓN REAL OUT-OF-SAMPLE DET: {iteracion}")
            print('='*50)
            print(f"  - Costo Real I  RP:       $ {costos_reales['Costo_Real_Almacenamiento_I_RP']:,.2f}")
            print(f"  - Costo Real II RP:       $ {costos_reales['Costo_Real_Maduracion_II_RP']:,.2f}")
            print(f"  - Costo Nerviosismo RP:   $ {costos_reales['Costo_Real_Nerviosismo_RP']:,.2f}")
            print(f"  - Costo Real III RP:      $ {costos_reales['Costo_Real_Stock_Terminado_III_RP']:,.2f}")
            print(f"  - Costo Real IV RP:       $ {costos_reales['Costo_Real_Backlog_IV_RP']:,.2f}")
            print(f" COSTO REAL TOTAL RP:       $ {costos_reales['Costo_Real_Total_RP']:,.2f}")
            print('='*50 + '\n')

            # --- EXPORTAR SOLUCIÓN OOS ---
            nombre_excel_oos = f"Resultados_RH_Det_OOS_Iteracion_{iteracion}.xlsx"
            print(f"Exportando resultados OOS de la iteración a: {nombre_excel_oos}")
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel_oos)
            
            # --- ACTUALIZAR ESTADO INICIAL ---
            self.Actualizar_Estado_Inicial(inst, q_plan_referencia=q_plan_referencia)
            self.Liberar_MPS_RP(inst)
            
        print("\nCiclo de Horizonte Móvil Determinista completado con éxito.")
        
        self.Exportar_Historial_Costos()
        self.Exportar_Historial_Costos_InSample()
            
if __name__ == "__main__":
    ruta_datos = "Datos_Originales.xlsx"
    ruta_pronosticos = "Evolucion_Pronosticos.xlsx"
    modelo_rh_det = PiscoRollingModel(
        data_path=ruta_datos, 
        pronostico_path=ruta_pronosticos,
        semilla_realidad=1001
    )
    
    modelo_rh_det.Ejecutar_Ciclo_Rolling_Horizon(iteraciones=4)
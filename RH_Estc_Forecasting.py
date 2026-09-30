import pyomo.environ as pyo
from pyomo.environ import value, minimize
import pandas as pd
import random
import numpy as np
from Pisco_Estc_Forecasting import PiscoModel 
from Integrar_Forecasting import generar_muestras_clark, cargar_demanda_real
from solver_utils import GurobiConfigurationError, create_gurobi_solver

class PiscoRollingModel:
    def __init__(self, data_path, pronostico_path, semilla):
        self.pronostico_path = pronostico_path
        self.semilla = semilla
        self.modelo_base = PiscoModel()
        self.modelo_base.ReadExcelFile(data_path)
        self.instance = None
        self.dia_calendario = 0
        #self.mes_calendario = 0
        self.alpha = 0.05 # Coeficiente de variabilidad de Clark
        self.historial_costos = []
        self.historial_costos_in_sample = []
        #self.demanda_real_dict = cargar_demanda_real(pronostico_path)
        self.rng = np.random.default_rng(semilla)

        # --- INTEGRACIÓN DE PRONÓSTICOS ---
        # Cargamos el archivo de pronósticos completo una sola vez.
        df_pronosticos_full = pd.read_excel(pronostico_path, engine='openpyxl')
        
        # Creamos un diccionario para búsqueda rápida con la clave (mezcla, dia_absoluto).
        # Guardamos tanto el pronóstico (F_t) como el lead time (t) original.
        self.pronosticos_dict = {}
        for _, row in df_pronosticos_full.iterrows():
            mezcla = row['n']
            fecha_hoy = int(row['Fecha de Hoy'])
            dia_entrega = int(fecha_hoy + row['t'])
            # CAMBIO OOS-VSS: la fecha del pronostico evita filtrar v0 de fechas futuras.
            self.pronosticos_dict[(mezcla, fecha_hoy, dia_entrega)] = {
                'F_t': max(0, row['F_t']),
                'lead_time': row['t']
            }

    def Construir_Modelo_Nerviosismo(self, RP=30, C_nerv_val=5.0):
        self.instance = self.modelo_base.Problema()
        
        inst = self.instance 

        # Longitud del periodo congelado RP = 30
        inst.RP = pyo.Param(initialize=RP, mutable=True)
        
        # Penalización económica por cambiar el plan de referencia
        inst.C_nerv = pyo.Param(initialize=C_nerv_val, mutable=True)
        
        
        # Plan de referencia (ayer
        # Se inicializa en 0, no hay nerviosismo en la iteración 0
        inst.q_bar = pyo.Param(inst.A, inst.Tp, initialize=0.0, mutable=True)

        # Variable auxiliar
        # f_at representa el valor absoluto de la desviación |q_at - q_bar_at|
        inst.f = pyo.Var(inst.A, inst.Tp, within=pyo.NonNegativeReals, initialize=0.0)

        # Retriccion de linealizacion
        # Se debe limitar la penalización al horizonte no congelado: t <= |T| - RP
        
        # f_at >= q_at - q_bar_at
        def lin_nerv_1_rule(model, a, t):
            if t > len(model.Tp) - pyo.value(model.RP):
                return pyo.Constraint.Skip
            return model.f[a, t] >= model.q[a, t] - model.q_bar[a, t]
        inst.const_nerv_1 = pyo.Constraint(inst.A, inst.Tp, rule=lin_nerv_1_rule)

        # f_at >= q_bar_at - q_at
        def lin_nerv_2_rule(model, a, t):
            if t > len(model.Tp) - pyo.value(model.RP):
                return pyo.Constraint.Skip
            return model.f[a, t] >= model.q_bar[a, t] - model.q[a, t]
        inst.const_nerv_2 = pyo.Constraint(inst.A, inst.Tp, rule=lin_nerv_2_rule)

        # Funcion objetivo actualizada con penalizacion por nerviosismo
        inst.FO.deactivate()
        def obj_rolling_rule(model):
            costo_original = model.FO.expr
            costo_nerviosismo = sum(model.C_nerv * model.f[a, t] for a in model.A
                for t in model.Tp if t <= len(model.Tp) - pyo.value(model.RP)) # Se aplica solo a t en {1, ..., |T| - RP}
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
        # CAMBIO OOS-VSS: q_bar futuro debe venir del plan a ciegas, no de la evaluacion con v0.
        return {(a, t): self._valor_variable(inst.q[a, t]) for a in inst.A for t in inst.Tp}

    def Seleccionar_Escenario_InSample(self, inst, iteracion):
        #rng = np.random.default_rng(self.semilla + self.dia_calendario)
        # CAMBIO OOS-VSS: mantiene visible la realidad simulada dentro de los escenarios SAA.
        escenarios = list(inst.S)
        probabilidades = [pyo.value(inst.pi[w]) for w in escenarios]
        escenario_real = self.rng.choice(escenarios, p=probabilidades)
        #escenario_real = random.choices(escenarios, weights=probabilidades, k=1)[0]
        print('\n' + '-'*50)
        print(f" ESCENARIO IN-SAMPLE SAA ITERACIÓN {iteracion}: {escenario_real}")
        print(f" Escenario elegido entre {len(escenarios)} escenarios de planificación.")
        print(" Luego se evaluará la misma decisión contra v0 (escenario OOS/101).")
        print('-'*50)
        return escenario_real

    def Congelar_MPS_RP(self, inst):
        # CAMBIO OOS-VSS: fija el MPS implementable antes de revelar v0.
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
        # CAMBIO OOS-VSS: devuelve grados de libertad al avanzar el horizonte.
        for n in inst.N:
            for t in self._periodos_rp(inst):
                inst.y[n, t].unfix()

        for a in inst.A:
            for t in self._periodos_rp(inst):
                inst.q[a, t].unfix()
                if hasattr(inst, 'l'):
                    inst.l[a, t].unfix()

   # def Inyectar_Demanda_Real_OutOfSample(self, inst):
   #     # CAMBIO OOS-VSS: v0 se replica en los 100 escenarios; conceptualmente es el escenario 101.
   #     print(f"Inyectando demanda real v0 para evaluar desde el día calendario: {self.dia_calendario}")
   #     for n in inst.N:
   #         n_name = self._obtener_nombre_mezcla(n)
   #         for t in inst.Tp:
   #             dia_real = self.dia_calendario + t
   #             demanda_real = self.demanda_real_dict.get((n_name, dia_real), 0.0)
   #             for w in inst.S:
   #                 inst.D_nts[n, t, w] = demanda_real

    def Inyectar_Demanda_Real_OutOfSample(self, inst, pronostico_path, semilla_101=1001):
        # Importamos el generador de la realidad sintética que creaste en Integrar_Forecasting
        from Integrar_Forecasting import cargar_escenario_101
        
        print(f"Inyectando Escenario 101 Sintético en todos los escenarios S para evaluar Día: {self.dia_calendario}")
        
        # 1. Generamos la demanda de la realidad sintética (Diccionario 2D: mezcla, dia_entrega)
        demanda_realidad_101 = cargar_escenario_101(
            ruta_csv=pronostico_path, 
            dia_actual=self.dia_calendario,
            semilla_realidad=semilla_101
        )
        
        # 2. Inyectamos el MISMO valor real en TODOS los escenarios del parámetro D_nts (3D)
        for n in inst.N:
            n_name = self._obtener_nombre_mezcla(n)
            for t in inst.Tp:
                dia_real = self.dia_calendario + t
                # Obtenemos el valor dictaminado por la realidad 101
                valor_realidad = demanda_realidad_101.get((n_name, dia_real), 0.0)
                
                # REEMPLAZO CLAVE: Sobrescribimos la incertidumbre. 
                # Ahora todos los escenarios 's' contienen la misma realidad.
                for s in inst.S:
                    inst.D_nts[n, t, s] = valor_realidad

    def Calcular_Costos_Reales_RP(self, inst):
        # CAMBIO OOS-VSS: costos realizados dentro del RP; con v0 replicado, el valor esperado es el costo real.
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
            pyo.value(inst.pi[s])
            * pyo.value(inst.Cwip_n[n, p])
            * self._valor_variable(inst.k[n, p, t, s])
            for s in inst.S
            for (n, p) in inst.Pn
            for t in periodos
        )
        costo_iv = sum(
            pyo.value(inst.pi[s]) * ci_minus * self._valor_variable(inst.i_minus[n, t, s])
            for s in inst.S
            for n in inst.N
            for t in periodos
        )
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

    def Exportar_Historial_Costos(self, nombre_archivo="Historial_Costos_RH_Estocastico_OOS.xlsx"):
        # CAMBIO OOS-VSS: historial completo de costos reales por iteracion.
        if not self.historial_costos:
            print("No hay historial de costos para exportar.")
            return

        df_historial = pd.DataFrame(self.historial_costos)
        print(f"\nExportando historial de costos a '{nombre_archivo}'...")
        df_historial.to_excel(nombre_archivo, index=False, engine='openpyxl')
        print("Exportación de historial completada.")

    def Exportar_Historial_Costos_InSample(self, nombre_archivo="Historial_Costos_RH_Estocastico_InSample.xlsx"):
        if not self.historial_costos_in_sample:
            print("No hay historial in-sample para exportar.")
            return

        df_historial = pd.DataFrame(self.historial_costos_in_sample)
        print(f"\nExportando historial in-sample a '{nombre_archivo}'...")
        df_historial.to_excel(nombre_archivo, index=False, engine='openpyxl')
        print("Exportación de historial in-sample completada.")

    def Actualizar_Estado_Inicial(self, inst, q_plan_referencia=None, usar_demanda_real_oos=False):
    # Obtenemos el valor escalar del periodo a congelar/avanzar
    # El estado del sistema en t = RP se extrae y las convierte 
    # en las nuevas condiciones iniciales para la sig iteracion
        RP = int(pyo.value(inst.RP))
        max_t = max(inst.Tp)
        
        escenarios = list(inst.S)
        if usar_demanda_real_oos:
            # CAMBIO OOS-VSS: v0 ya fue replicado en S; cualquier escenario representa la realidad 101.
            escenario_real = escenarios[0]
            print(f" Estado actualizado con realidad 101 replicada en S. Escenario base: '{escenario_real}'")
        else:
            # simulación realidad in-sample: se sortea uno de los escenarios del SAA.
            probabilidades = [pyo.value(inst.pi[w]) for w in escenarios]
            escenario_real = random.choices(escenarios, weights=probabilidades, k=1)[0]
            print(f" Escenario:'{escenario_real}'")
        
        # inventario y llegadas de alcohol
        for a in inst.A:
            # El inventario físico al final del día RP será el inventario inicial de mañana (I_a)
            inst.I_a[a] = pyo.value(inst.v[a, RP])
            
            # Los camiones que llegaron en t=RP serán las llegadas iniciales (omega_a) 
            inst.W_a[a] = pyo.value(inst.q[a, RP])
            
        # Backlogs - Estoc
        for n in inst.N:
            # Si quedamos debiendo producto en el día RP, esa es nuestra nueva deuda inicial (I_minus_n)
            # Usamos el valor de i_minus solo para el escenario que ocurrió realmente
            inst.I_minus_n[n] = pyo.value(inst.i_minus[n, RP, escenario_real])
        
        # Inventario en proceso - WIP
        for (n, p) in inst.Pn:
            tau_val = int(pyo.value(inst.Tau_np[n, p]))
            for u in inst.U:
                if u < tau_val:
                    # El líquido que alcanzó la edad 'u' en el día RP, inicia el nuevo horizonte con esa edad (gamma_nup)
                    inst.gamma[n, p, u] = pyo.value(inst.z[n, p, RP, u])
                    
        # INVENTARIO LISTO INTERMEDIO (GRUPOS G) - Estoc
        # Recalculamos I_Gp sumando el stock listo (k_npt) de las mezclas que pertenecen al grupo.
        for (g_name, p_id) in inst.Set_Grupos_P:
            lista_mezclas_ids = self.modelo_base.Group_Mapping.get((g_name, p_id), [])
            
            # Sumamos el inventario listo de todas las mezclas del grupo en t=RP
            # Solo para el escenario que ocurrió realmente
            suma_k = sum(pyo.value(inst.k[n_id, p_id, RP, escenario_real]) for n_id in lista_mezclas_ids)
            inst.I_Gp[g_name, p_id] = suma_k

        # Rolling, demandas y referencias
        # Actualizar plan de referencia para el nerviosismo (q_bar)        
        # 1.Actualizar plan de referencia para el nerviosismo (q_bar)
        for a in inst.A:
            for t in inst.Tp:
                if t + RP <= max_t:
                    # Lo planeado para el día t+RP se desplaza al día t
                    if q_plan_referencia is not None:
                        inst.q_bar[a, t] = q_plan_referencia.get((a, t + RP), 0.0)
                    else:
                        inst.q_bar[a, t] = pyo.value(inst.q[a, t + RP])
                else:
                    # Al final del horizonte no tenemos datos previos, asumimos 0
                    inst.q_bar[a, t] = 0.0
                    
        # Sumamos los días que avanzamos al calendario 
        #self.mes_calendario += RP
        self.dia_calendario += RP
        
        # --- ACTUALIZACIÓN DE ESCENARIOS DE DEMANDA ---
        print(f"Actualizando escenarios de demanda para el día calendario: {self.dia_calendario}")
        #rng = np.random.default_rng(42 + self.dia_calendario) # Reproducible y distinta por iteracion
        rng = self.rng # Reproducible y distinta por iteracion
        for n in inst.N:
            n_name = self.modelo_base.Mezclas.iloc[n-1]['n']
            for t in inst.Tp:
                dia_real = self.dia_calendario + t

                forecast_data = self.pronosticos_dict.get((n_name, self.dia_calendario, dia_real))

                if forecast_data:
                    media = forecast_data['F_t']
                    lead_time = forecast_data['lead_time']

                    # Generamos escenarios con convergencia no lineal sqrt(log(t + 1))
                    # y dispersion limitada a +/- 4 sigma.
                    muestras = generar_muestras_clark(
                        media=media,
                        tiempo_restante=lead_time,
                        num_muestras=len(inst.S),
                        alpha=self.alpha,
                        rng=rng
                    )

                    for idx, w in enumerate(inst.S):
                        inst.D_nts[n, t, w] = muestras[idx]
                else:
                    # Si no hay pronóstico, la demanda es 0 en todos los escenarios
                    for w in inst.S:
                        inst.D_nts[n, t, w] = 0.0

        print("Actualización completada.")

    def Ejecutar_Ciclo_Rolling_Horizon(self, iteraciones=4):
        
        #random.seed(42)
        rng = np.random.default_rng(self.semilla + self.dia_calendario)
        # Cargar los escenarios iniciales desde el archivo de pronósticos
        self.modelo_base.Cargar_Escenarios_Forecasting(
            ruta_csv=self.pronostico_path,
            dia_actual=0,
            num_escenarios=100,
            alpha=self.alpha
        )
        
        # RP = 1 periodo (30 dias)
        inst = self.Construir_Modelo_Nerviosismo(RP=30, C_nerv_val=5.0)
        try:
            solver = create_gurobi_solver({"TimeLimit": 1800, "MIPGap": 0.05, "Threads": 1})
        except GurobiConfigurationError as exc:
            print(f"\nError de configuración de Gurobi:\n{exc}\n")
            return

        for iteracion in range(iteraciones):
        #Bucle principal para rodar el horizonte de planificación y exportar resultados.
            if iteracion == 0:
                inst.C_nerv.value = 0.0  # La corrida 0 es estática, no hay plan previo
            else:
                inst.C_nerv.value = 5.0 # A partir de la iter 1, penalizacion por los cambios
            
            # CAMBIO OOS-VSS: Fase 1, planificacion a ciegas con los 100 escenarios SAA.
            results = solver.solve(inst, tee=False)
            
            if self._solucion_exitosa(results):
                
                # Costo de Nerviosismo
                RP_val = pyo.value(inst.RP)
                c_nerv_val = pyo.value(inst.C_nerv)
                costo_nerv_iter = sum(c_nerv_val * pyo.value(inst.f[a, t]) 
                                      for a in inst.A for t in inst.Tp 
                                      if t <= len(inst.Tp) - RP_val)

                print(f'-'*50)
                print(f" Solución Iteración:              {iteracion:.2f}")
                print(' 1️⃣ COSTOS DE PRIMERA ETAPA (Decisiones Fijas):')
                print(f'  - Costo I (Almacenamiento):         $ {pyo.value(inst.coste_I):,.2f}')
                print(f'  - Costo II (Inventario Maduración): $ {pyo.value(inst.coste_II):,.2f}')
                print(f'  - Costo de Nerviosismo (Cambios):   $ {costo_nerv_iter:,.2f}') # AQUÍ SE IMPRIME
                
                # Sumamos el nerviosismo al total de Primera Etapa
                total_1ra = pyo.value(inst.coste_I) + pyo.value(inst.coste_II) + costo_nerv_iter
                print(f"Costo 1ra Etapa Total: $ {total_1ra:,.2f}")
                print('-'*50)
                print(' 2️⃣ COSTOS DE SEGUNDA ETAPA (Valor Esperado):')
                print(f'  - Costo III (Stock Terminado):      $ {pyo.value(inst.coste_III):,.2f}')
                print(f'  - Costo IV (Penalización Backlog):  $ {pyo.value(inst.coste_IV):,.2f}')
                total_2da = pyo.value(inst.coste_III) + pyo.value(inst.coste_IV)
                print(f"Costo 2da Etapa Total: $ {total_2da:,.2f}")
                print('-'*50)
                print(f"Costo Total FO (Rolling):       $ {pyo.value(inst.FO_Rolling):,.2f}")
                print('-'*50)
            else:
                print(f"Error: Solver no convergió. Estado: {results.solver.termination_condition}")
                break

            costo_planificacion_total = pyo.value(inst.FO_Rolling)
            q_plan_referencia = self.Guardar_Plan_Q(inst)

            # CAMBIO OOS-VSS: se informa el escenario in-sample elegido antes de pasar a v0.
            escenario_insample = self.Seleccionar_Escenario_InSample(inst, iteracion)

            # Exporta la solución in-sample con el mismo formato que la evaluación OOS
            nombre_excel_is = f"Resultados_RH_Estc_IS_Iteracion_{iteracion}.xlsx"
            print(f"Exportando resultados in-sample de la iteración a: {nombre_excel_is}")
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel_is)

            costo_nerv_iter = sum(
                pyo.value(inst.C_nerv) * pyo.value(inst.f[a, t])
                for a in inst.A
                for t in inst.Tp
                if t <= len(inst.Tp) - int(pyo.value(inst.RP))
            )

            fila_in_sample = {
                "Iteracion": iteracion,
                "Dia_Inicio": self.dia_calendario,
                "Dia_Fin_RP": self.dia_calendario + int(pyo.value(inst.RP)),
                "Escenario_InSample_Sorteado": escenario_insample,
                "Costo_I": float(pyo.value(inst.coste_I)),
                "Costo_II": float(pyo.value(inst.coste_II)),
                "Costo_Nerviosismo": float(costo_nerv_iter),
                "Costo_III": float(pyo.value(inst.coste_III)),
                "Costo_IV": float(pyo.value(inst.coste_IV)),
                "Costo_Total_InSample": float(pyo.value(inst.FO_Rolling)),
                "Escenario_Evaluacion": "InSample_SAA",
                "Semilla": getattr(self, "semilla", 61),
            }
            self.historial_costos_in_sample.append(fila_in_sample)

            # CAMBIO OOS-VSS: Fase 2, congelar MPS e inyectar v0 como realidad externa.
            self.Congelar_MPS_RP(inst)
            #self.Inyectar_Demanda_Real_OutOfSample(inst)
            self.Inyectar_Demanda_Real_OutOfSample(inst, pronostico_path="Evolucion_Pronosticos.xlsx")
            results_real = solver.solve(inst, tee=False)

            if not self._solucion_exitosa(results_real):
                print(f"Error: evaluación real no convergió. Estado: {results_real.solver.termination_condition}")
                self.Liberar_MPS_RP(inst)
                break

            costos_reales = self.Calcular_Costos_Reales_RP(inst)
            datos_iteracion = {
                'Iteracion': iteracion,
                'Modelo': 'Estocastico',
                'Dia_Inicio': self.dia_calendario,
                'Dia_Fin_RP': self.dia_calendario + int(pyo.value(inst.RP)),
                'Escenario_InSample_Sorteado': escenario_insample,
                'Escenario_Evaluacion_OOS': 'Escenario_101',
                'Costo_Planificacion_Total_HL': costo_planificacion_total,
            }
            datos_iteracion.update(costos_reales)
            self.historial_costos.append(datos_iteracion)

            print('\n' + '='*50)
            print(f" EVALUACIÓN REAL OUT-OF-SAMPLE ESTC: {iteracion}")
            print('='*50)
            print(f"  - Costo Real I  RP:       $ {costos_reales['Costo_Real_Almacenamiento_I_RP']:,.2f}")
            print(f"  - Costo Real II RP:       $ {costos_reales['Costo_Real_Maduracion_II_RP']:,.2f}")
            print(f"  - Costo Nerviosismo RP:   $ {costos_reales['Costo_Real_Nerviosismo_RP']:,.2f}")
            print(f"  - Costo Real III RP:      $ {costos_reales['Costo_Real_Stock_Terminado_III_RP']:,.2f}")
            print(f"  - Costo Real IV RP:       $ {costos_reales['Costo_Real_Backlog_IV_RP']:,.2f}")
            print(f" COSTO REAL TOTAL RP:       $ {costos_reales['Costo_Real_Total_RP']:,.2f}")
            print('='*50 + '\n')

            #  EXPORTAR RESULTADOS 
            nombre_excel = f"Resultados_RH_Estc_OOS_Iteracion_{iteracion}.xlsx"
            print(f"Exportando resultados de la iteración a: {nombre_excel}")
            # Llamamos a la función original intacta
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel)
            
            # ACTUALIZAR CONDICIONES INICIALES (RODAR HORIZONTE)
        
            self.Actualizar_Estado_Inicial(
                inst,
                q_plan_referencia=q_plan_referencia,
                usar_demanda_real_oos=True,
            )
            self.Liberar_MPS_RP(inst)
            
        print("\nCiclo de Horizonte Móvil completado con éxito.")
        self.Exportar_Historial_Costos()
        self.Exportar_Historial_Costos_InSample()

if __name__ == "__main__":
    ruta_datos = "Datos_Originales.xlsx"
    ruta_pronosticos = "Evolucion_Pronosticos.xlsx"
    modelo_rh = PiscoRollingModel(data_path=ruta_datos, pronostico_path=ruta_pronosticos, semilla=61)
    
    # Probamos el ciclo del horizonte móvil
    modelo_rh.Ejecutar_Ciclo_Rolling_Horizon(iteraciones=4)
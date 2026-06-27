import pyomo.environ as pyo
from pyomo.environ import value, minimize
import pandas as pd

from PiscoDet import PiscoModel_Operativo
from solver_utils import GurobiConfigurationError, create_gurobi_solver

class PiscoRollingModel:
    def __init__(self, data_path):
        
        self.modelo_base = PiscoModel_Operativo()
        self.modelo_base.ReadExcelFile(data_path)
        self.instance = None
        self.dia_calendario = 0
        
    def Construir_Modelo_Nerviosismo(self, RP=30, C_nerv_val=5.0):

        self.instance = self.modelo_base.Problema()
        inst = self.instance 
        
        # Parametros 
        # Longitud del periodo congelado RP
        inst.RP = pyo.Param(initialize=RP, mutable=True)
        
        # Penalización económica por cambiar el plan de recepción
        inst.C_nerv = pyo.Param(initialize=C_nerv_val, mutable=True)
        
        # Plan de referencia (ayer)
        # Se inicializa en 0, no hay nerviosismo en la iteración 0
        inst.q_bar = pyo.Param(inst.A, inst.Tp, initialize=0.0, mutable=True)

        # Variable auxiliar
        # f_at va a representar el valor absoluto de la desviación |q_at - q_bar_at|
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
    
    def Actualizar_Estado_Inicial(self, inst):
        # Obtenemos el valor escalar del periodo a congelar/avanzar
        # El estado del sistema en t = RP se extrae y las convierte en las nuevas condiciones iniciales para la sig iteracion
        RP = int(pyo.value(inst.RP))
        max_t = max(inst.Tp)
        
        # inventario y llegadas de alcohol
        for a in inst.A:
            # El inventario físico al final del día RP será el inventario inicial de mañana (I_a)
            inst.I_a[a] = pyo.value(inst.v[a, RP])
            
            # Los camiones que llegaron en t=RP serán las llegadas iniciales (omega_a) 
            inst.W_a[a] = pyo.value(inst.q[a, RP]) # <--- SOLUCIÓN (Usar W_a)            
            
        # backlogs
        for n in inst.N:
            # Si se queda debiendo producto en el día RP, esa es nuestra nueva deuda inicial (I_minus_n)
            inst.I_minus_n[n] = pyo.value(inst.i_minus[n, RP])
            
        # Inventario en proceso - WIP
        for (n, p) in inst.Pn:
            tau_val = int(pyo.value(inst.Tau_np[n, p]))
            for u in inst.U:
                if u < tau_val:
                    # El líquido que alcanzó la edad u en el día RP, inicia el nuevo horizonte con esa edad (gamma_nup)
                    inst.gamma[n, p, u] = pyo.value(inst.z[n, p, RP, u])
                    
        # inventario grupos
        # Recalculamos I_Gp sumando el stock listo (k_npt) de las mezclas que pertenecen al grupo.
        for (g_name, p_id) in inst.Set_Grupos_P:
            lista_mezclas_ids = self.modelo_base.Group_Mapping.get((g_name, p_id), [])
            
            # Sumamos el inventario listo de todas las mezclas del grupo en t=RP
            suma_k = sum(pyo.value(inst.k[n_id, p_id, RP]) for n_id in lista_mezclas_ids)
            inst.I_Gp[g_name, p_id] = suma_k

        # Rolling, demandas y referencias
        # Actualizar plan de referencia para el nerviosismo (q_bar)
        for a in inst.A:
            for t in inst.Tp:
                if t + RP <= max_t:
                    # Lo planeado para el día t+RP se desplaza al día t
                    inst.q_bar[a, t] = pyo.value(inst.q[a, t + RP])
                else:
                    # Al final del horizonte no tenemos datos previos, asumimos 0
                    inst.q_bar[a, t] = 0.0
                    
        # Sumamos los días que avanzamos
        self.dia_calendario += RP
        
        # Actualizar Demanda (D_nt) leyendo directamente del excel
        for n in inst.N:
            n_name = self.modelo_base.Mezclas.iloc[n-1]['n']
            for t in inst.Tp:
                # Calculamos qué día real del mes t relativo
                dia_real = self.dia_calendario + t
                
                if (n_name, dia_real) in self.modelo_base.Demands.index:
                    nueva_demanda = self.modelo_base.Demands.loc[(n_name, dia_real), 'D']
                    inst.D_nt[n, t] = float(nueva_demanda)
                else:
                    inst.D_nt[n, t] = 0.0

    def Ejecutar_Ciclo_Rolling_Horizon(self, iteraciones=12):

        # Bucle principal para rodar el horizonte de planificación y exportar resultados.
        inst = self.Construir_Modelo_Nerviosismo(RP=30, C_nerv_val=5.0)
        try:
            solver = create_gurobi_solver({"TimeLimit": 900, "MIPGap": 0})
        except GurobiConfigurationError as exc:
            print(f"\nError de configuración de Gurobi:\n{exc}\n")
            return

        for iteracion in range(iteraciones):
            if iteracion == 0:
                inst.C_nerv.value = 0.0  # Corrida inicial, sin penalización previa
            else:
                inst.C_nerv.value = 5.0 # Se activa la penalización
                
            results = solver.solve(inst, tee=False) 
            
            # Condición de término flexibilizada (Optimal o TimeLimit)
            if (results.solver.status == pyo.SolverStatus.ok) and \
               (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit]):
                
                # Cálculo de Nerviosismo al vuelo
                RP_val = pyo.value(inst.RP)
                c_nerv_val = pyo.value(inst.C_nerv)
                costo_nerv_iter = sum(c_nerv_val * pyo.value(inst.f[a, t]) 
                                      for a in inst.A for t in inst.Tp 
                                      if t <= len(inst.Tp) - RP_val)

                print(f'\n' + '='*50)
                print(f" ITERACIÓN ROLLING HORIZON:       {iteracion}")
                print('='*50)
                print(' 1️⃣ COSTOS OPERATIVOS (Primera Etapa / Fijos):')
                print(f'  - Costo I (Almacenamiento):         $ {pyo.value(inst.coste_I):,.2f}')
                print(f'  - Costo II (Inventario Maduración): $ {pyo.value(inst.coste_II):,.2f}')
                print(f'  - Costo de Nerviosismo (Cambios):   $ {costo_nerv_iter:,.2f}')
                
                total_1ra = pyo.value(inst.coste_I) + pyo.value(inst.coste_II) + costo_nerv_iter
                print(f" Subtotal Operativo:                  $ {total_1ra:,.2f}")
                print('-'*50)
                print(' 2️⃣ COSTOS DE STOCK Y PENALIZACIÓN:')
                print(f'  - Costo III (Stock Terminado):      $ {pyo.value(inst.coste_III):,.2f}')
                print(f'  - Costo IV (Multas Backlog):        $ {pyo.value(inst.coste_IV):,.2f}')
                total_2da = pyo.value(inst.coste_III) + pyo.value(inst.coste_IV)
                print(f" Subtotal Penalizaciones y Stock:     $ {total_2da:,.2f}")
                print('-'*50)
                print(f" COSTO TOTAL FO (Rolling):            $ {pyo.value(inst.FO_Rolling):,.2f}")
                print('='*50 + '\n')
            else:
                print(f"Error: Solver no convergió. Estado: {results.solver.termination_condition}")
                break

            # EXPORTAR RESULTADOS 
            nombre_excel = f"Resultados_RH_Det_Iteracion_{iteracion}.xlsx"
            print(f"Exportando resultados de la iteración a: {nombre_excel}")
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel)
            
            # ACTUALIZAR CONDICIONES INICIALES Y RODAR HORIZONTE
            self.Actualizar_Estado_Inicial(inst)
            
        print("\nCiclo de Horizonte Móvil Determinista completado con éxito.")
            
if __name__ == "__main__":
    ruta_datos = "Datos_Originales.xlsx"
    modelo_rh_det = PiscoRollingModel(ruta_datos)
    
    # Ejecutamos el experimento para 4 iteraciones (120 días)
    modelo_rh_det.Ejecutar_Ciclo_Rolling_Horizon(iteraciones=12)

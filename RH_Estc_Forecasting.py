import pyomo.environ as pyo
from pyomo.environ import value, minimize
import pandas as pd
import random
import numpy as np
from Pisco_Estc_Forecasting import PiscoModel 
from Integrar_Forecasting import cargar_demanda_estocastica
from solver_utils import GurobiConfigurationError, create_gurobi_solver

class PiscoRollingModel:
    def __init__(self, data_path, pronostico_path):
        self.modelo_base = PiscoModel()
        self.modelo_base.ReadExcelFile(data_path)
        self.instance = None
        self.dia_calendario = 0
        #self.mes_calendario = 0
        self.alpha = 0.05 # Coeficiente de variabilidad de Clark

        # --- INTEGRACIÓN DE PRONÓSTICOS ---
        # Cargamos el archivo de pronósticos completo una sola vez.
        df_pronosticos_full = pd.read_excel(pronostico_path, engine='openpyxl')
        
        # Creamos un diccionario para búsqueda rápida con la clave (mezcla, dia_absoluto).
        # Guardamos tanto el pronóstico (F_t) como el lead time (t) original.
        self.pronosticos_dict = {}
        for _, row in df_pronosticos_full.iterrows():
            mezcla = row['n']
            dia_entrega = int(row['Fecha de Hoy'] + row['t'])
            self.pronosticos_dict[(mezcla, dia_entrega)] = {
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
    
    def Actualizar_Estado_Inicial(self, inst):
    # Obtenemos el valor escalar del periodo a congelar/avanzar
    # El estado del sistema en t = RP se extrae y las convierte 
    # en las nuevas condiciones iniciales para la sig iteracion
        RP = int(pyo.value(inst.RP))
        max_t = max(inst.Tp)
        
        # simulación realidad
        # decisión al azar basados en las probabilidades pi_s del modelo base.
        # Extraemos los nombres de los escenarios y sus probabilidades de la instancia
        escenarios = list(inst.S)
        probabilidades = [pyo.value(inst.pi[w]) for w in escenarios]
        
        # Elegimos aleatoriamente qué escenario ocurrió realmente en este día RP
        # 1 de los 50 escenarios
        # random.choices devuelve una lista, tomamos el primer elemento [0]
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
                    inst.q_bar[a, t] = pyo.value(inst.q[a, t + RP])
                else:
                    # Al final del horizonte no tenemos datos previos, asumimos 0
                    inst.q_bar[a, t] = 0.0
                    
        # Sumamos los días que avanzamos al calendario 
        #self.mes_calendario += RP
        self.dia_calendario += RP
        
        # --- ACTUALIZACIÓN DE ESCENARIOS DE DEMANDA ---
        print(f"Actualizando escenarios de demanda para el día calendario: {self.dia_calendario}")
        np.random.seed(42 + self.dia_calendario) # Semilla reproducible pero diferente para cada iteración

        for n in inst.N:
            n_name = self.modelo_base.Mezclas.iloc[n-1]['n']
            for t in inst.Tp:
                dia_real = self.dia_calendario + t

                forecast_data = self.pronosticos_dict.get((n_name, dia_real))

                if forecast_data:
                    media = forecast_data['F_t']
                    lead_time = forecast_data['lead_time']

                    # La desviación estándar se basa en el modelo de Clark
                    desviacion_est = abs(media * self.alpha * lead_time)

                    # Generamos nuevos escenarios
                    muestras = np.random.normal(loc=media, scale=desviacion_est, size=len(inst.S))

                    for idx, w in enumerate(inst.S):
                        inst.D_nts[n, t, w] = max(0.0, muestras[idx])
                else:
                    # Si no hay pronóstico, la demanda es 0 en todos los escenarios
                    for w in inst.S:
                        inst.D_nts[n, t, w] = 0.0

        print("Actualización completada.")

    def Ejecutar_Ciclo_Rolling_Horizon(self, iteraciones=1):
        
        random.seed(42)
        # Cargar los escenarios iniciales desde el archivo de pronósticos
        self.modelo_base.Cargar_Escenarios_Forecasting(
            ruta_csv="Evolucion_Pronosticos.xlsx",
            dia_actual=0,
            num_escenarios=100,
            alpha=self.alpha
        )
        
        # RP = 1 periodo (30 dias)
        inst = self.Construir_Modelo_Nerviosismo(RP=30, C_nerv_val=5.0)
        try:
            solver = create_gurobi_solver({"TimeLimit": 1800, "MIPGap": 0.001})
        except GurobiConfigurationError as exc:
            print(f"\nError de configuración de Gurobi:\n{exc}\n")
            return

        for iteracion in range(iteraciones):
        #Bucle principal para rodar el horizonte de planificación y exportar resultados.
            if iteracion == 0:
                inst.C_nerv = 0.0  # La corrida 0 es estática, no hay plan previo
            else:
                inst.C_nerv.value = 5.0 # A partir de la iter 1, penalizacion por los cambios
            
            # Resolver el modelo
            results = solver.solve(inst, tee=False)
            
            if (results.solver.status == pyo.SolverStatus.ok) and \
               (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit]):
                
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

            #  EXPORTAR RESULTADOS 
            nombre_excel = f"Resultados_RH_Periodo_{iteracion}.xlsx"
            print(f"Exportando resultados de la iteración a: {nombre_excel}")
            # Llamamos a la función original intacta
            self.modelo_base.ExportarResultados(inst, nombre_archivo=nombre_excel)
            
            # ACTUALIZAR CONDICIONES INICIALES (RODAR HORIZONTE)
        
            self.Actualizar_Estado_Inicial(inst)
            
        print("\nCiclo de Horizonte Móvil completado con éxito.")
            
if __name__ == "__main__":
    ruta_datos = "Datos_Originales.xlsx"
    ruta_pronosticos = "Evolucion_Pronosticos.xlsx"
    modelo_rh = PiscoRollingModel(data_path=ruta_datos, pronostico_path=ruta_pronosticos)
    
    # Probamos el ciclo del horizonte móvil
    modelo_rh.Ejecutar_Ciclo_Rolling_Horizon(iteraciones=1)

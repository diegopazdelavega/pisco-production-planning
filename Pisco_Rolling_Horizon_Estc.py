import pyomo.environ as pyo
from pyomo.environ import value, minimize, SolverFactory
import pandas as pd
import random
import numpy as np

from PiscoV3_Estc import PiscoModel 

class PiscoRollingModel:
    def __init__(self, data_path):
        self.modelo_base = PiscoModel()
        self.modelo_base.ReadExcelFile(data_path)
        self.instance = None
        self.dia_calendario = 0
        #self.mes_calendario = 0
        

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

        
        # 2.Actualizar Demanda (D_nt_base) SAA Dinamico
        for n in inst.N:
            n_name = self.modelo_base.Mezclas.iloc[n-1]['n']
            for t in inst.Tp:
                # Calculamos qué día real del mes t relativo
                # en la iter 2, self.dia_calendario = 2
                # cuando t = 10 es el día 12 (2 + 10)
                dia_real = self.dia_calendario + t
                #mes_real = self.mes_calendario + t
                
                val_base = 0.0
                # pronostico Ft_nt' (dia 12)
                if (n_name, dia_real) in self.modelo_base.Demands.index:
                    val_base = float(self.modelo_base.Demands.loc[(n_name, dia_real), 'D'])
                
                # variabilidad del 20%
                desviacion = val_base * 0.20
                
                # RECONSTRUIMOS LOS 50 ESCENARIOS ESTADÍSTICOS
                if val_base == 0:
                    for w in inst.S:
                        inst.D_nts[n, t, w] = 0.0
                else:
                    # Generamos 50 nuevas muestras desde la campana de Gauss
                    muestras = np.random.normal(loc=val_base, scale=desviacion, size=len(inst.S))
                    for idx, w in enumerate(inst.S):
                        # Asignamos la muestra al escenario (evitando demandas negativas)
                        inst.D_nts[n, t, w] = max(0.0, muestras[idx])
                
                # # RECONSTRUIMOS LOS ESCENARIOS para los nuevos días que entraron al horizonte
                # for w in inst.S:
                #     if w == 'Low': 
                #         inst.D_nts[n, t, w] = val_base * 0.8
                #     elif w == 'High': 
                #         inst.D_nts[n, t, w] = val_base * 1.2
                #     else: 
                #         inst.D_nts[n, t, w] = val_base

        print("Actualización completada.")

    def Ejecutar_Ciclo_Rolling_Horizon(self, iteraciones=4):
        
        random.seed(42)
        # Generar los escenarios iniciales antes del modelo
        self.modelo_base.Generar_Escenarios_SAA(num_escenarios=10, variabilidad=0.20)
        
        # RP = 1 periodo (30 dias)
        inst = self.Construir_Modelo_Nerviosismo(RP=30, C_nerv_val=5.0)
        solver = pyo.SolverFactory('gurobi')
        
        solver.options['TimeLimit'] = 1800  # subir a 5 min
        solver.options['MIPGap'] = 0.06   # Aceptar un 5% de gap de optimidad

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
    ruta_datos = r"/Users/diegopazdelavega/Documents/Proyecto/capel/Codigos/Datos_Originales.xlsx"
    modelo_rh = PiscoRollingModel(ruta_datos)
    
    # Probamos el ciclo del horizonte móvil
    modelo_rh.Ejecutar_Ciclo_Rolling_Horizon(iteraciones=8)
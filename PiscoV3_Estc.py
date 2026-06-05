import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Set, Param, Var, Constraint, Objective, 
    Binary, NonNegativeReals, NonNegativeIntegers, minimize, value, SolverFactory,
    Expression
)

class PiscoModel: 
    def __init__(self, name=None):
        self.data = {}
        # Datos Estocásticos
        # self.data = ['Escenarios']
        # self.Escenarios = {'Low': 0.2, 'Base': 0.5, 'High': 0.3} # Diccionario {nombre: probabilidad}
        
        self.Ci_minus_val = 100       # Valor por defecto
        self.Cap_factor = 1.0         # 1.0 = 100% de la capacidad
        
    def ReadExcelFile(self, FileName):    
        try:
            self.data = pd.read_excel(FileName, sheet_name=None)
        except FileNotFoundError:
            return

        #limpieza de nombres de columnas
        for sheet, df in self.data.items():
            df.columns = df.columns.str.strip()

        self.Tiempo = self.data['Set_T']
        self.Alcoholes = self.data['Set_A']
        self.Mezclas = self.data['Set_N']
        self.Procesos = self.data['Set_P']
        self.Scalars = self.data['Scalars']
        self.Routes = self.data['Routes_Taus']  # Columnas: n, p, tau, Cwip
        self.Recipes = self.data['Recipes']     # Columnas: n, a, R
        self.Demands = self.data['Demands']     # Columnas: n, t, D
        self.Capacities = self.data['Capacities'].set_index('p')  # indexado pq no se itera capacities
        
        self.N_Tiempo = len(self.Tiempo)
        self.N_Alcoholes = len(self.Alcoholes)
        self.N_Mezclas = len(self.Mezclas)
        self.N_Procesos = len(self.Procesos)

        # mapeos, nombres a id
        self.map_n = {nombre: i+1 for i, nombre in enumerate(self.Mezclas['n'])}
        self.map_p = {nombre: i+1 for i, nombre in enumerate(self.Procesos['p'])}
        self.map_a = {nombre: i+1 for i, nombre in enumerate(self.Alcoholes['a'])}

        # Diccionarios de Sets
        self.Pn_dict = {i+1: [] for i in range(len(self.Mezclas))}
        for index, row in self.Routes.iterrows():
            if row['n'] in self.map_n and row['p'] in self.map_p:
                if row['p'] == 'reception':                                     # las mezclas no "pasan" por recepcion
                    continue
                self.Pn_dict[self.map_n[row['n']]].append(self.map_p[row['p']])
        
        self.An_dict = {i+1: [] for i in range(len(self.Mezclas))}
        for index, row in self.Recipes.iterrows():
            if row['n'] in self.map_n and row['a'] in self.map_a:
                self.An_dict[self.map_n[row['n']]].append(self.map_a[row['a']])

        # Índices para busqueda rápida, establece relaciones ej. (n,p) para Cwip
        self.Routes = self.Routes.set_index(['n', 'p'])
        self.Recipes = self.Recipes.set_index(['n', 'a'])
        self.Demands = self.Demands.set_index(['n', 't'])

        # ids procesos claves
        self.name_recep = 'reception'
        self.name_mix = 'mixing'
        self.name_ref = 'refinement'
        
        # Extracción Condiciones Iniciales
        # Inventario de alcoholes - Llegadas programadas alcoholes - 
        self.Init_Alcoholes = self.data['Inv_Inicial_Alcoholes'].set_index('a')  # Columnas: a, Ia
        self.Init_Omega = self.data['LLegadas_Alcoholes_Omega'].set_index('a')     # Columnas: a, omega
        self.Init_Backlog = self.data['Backlog_Inicial'].set_index('n')          # Columnas: n, I_minus_n
        self.Init_Gamma = self.data['Inventario_Antiguo_Gamma'].set_index(['n','p','u']) #Columnas: n, p, u, gamma_nup    
        
        # Condiciones Iniciales Grupos
        # LEER TABLAS EN FORMATO PAPER
        # El objetivo de este bloque es crear un diccionario llamado self.Group_Mapping 
        # que le dice al modelo qué mezclas exactas conforman un grupo en un proceso específico.
        
        self.Group_Mapping = {} 
        self.Group_Inv_Data = {}

        #  Asignacion de Grupos (Tabla B.6)
    
        if 'Asignacion_de_Grupos_N(p)' in self.data:
            df_map = self.data['Asignacion_de_Grupos_N(p)']
            
            # Iteramos por cada fila (Cada Grupo A1, A2...)
            # Lectura y recorrido de filas: for index, row in df_map.iterrows():
            # El código toma la tabla y la lee fila por fila. En cada fila, guarda el nombre del grupo en la variable g_name (por ejemplo, "A1").
            for index, row in df_map.iterrows():
                g_name = row['Group']  # La primera columna debe ser el nombre del grupo
                
                # Iteramos por las columnas que corresponden a procesos
                for col_name in df_map.columns:
                    if col_name == 'Group': continue # Saltamos la columna del nombre
                    
                    # Intentamos adivinar el proceso basado en el nombre de la columna
                    # Ej: Si columna es "N(mixing)", buscamos "mixing" en nuestros procesos
                    p_encontrado = None
                    for p_real in self.map_p.keys():
                        if p_real in col_name: # Si 'mixing' está en 'N(mixing)'
                            p_encontrado = p_real
                            break
                    
                    if p_encontrado:
                        celda = str(row[col_name]) # Convertimos a texto por seguridad
                        
                        # Si la celda tiene un guión o está vacía, saltamos
                        if celda == '-' or pd.isna(row[col_name]) or celda.strip() == '':
                            continue
                            
                        # Separamos por comas (para casos como "n4, n5")
                        mezclas_en_celda = [m.strip() for m in celda.split(',')]
                        
                        p_id = self.map_p[p_encontrado]
                        #crea una "llave" única combinando el Grupo y el Proceso: key = (g_name, p_id) (ej. "A1 en Estabilización"). Luego, guarda dentro de esa llave los IDs de las mezclas que encontró.
                        key = (g_name, p_id)
                        
                        if key not in self.Group_Mapping:
                            self.Group_Mapping[key] = []
                            
                        for m_nombre in mezclas_en_celda:
                            if m_nombre in self.map_n:
                                self.Group_Mapping[key].append(self.map_n[m_nombre])
                                # Un diccionario que se ve así: self.Group_Mapping = {('A1', ID_Estabilizacion): [ID_n1, ID_n3]}.

        #  Inventario de Grupos (Tabla B.7)
        
        if 'Inv_Inicial_Grupos_N(p)' in self.data:
            df_inv = self.data['Inv_Inicial_Grupos_N(p)']
            
            for index, row in df_inv.iterrows():
                g_name = row['Group']
                
                for col_name in df_inv.columns:
                    if col_name == 'Group': continue
                    
                    p_encontrado = None
                    for p_real in self.map_p.keys():
                        if p_real in col_name:
                            p_encontrado = p_real
                            break
                    
                    if p_encontrado:
                        valor = row[col_name]
                        
                        # Validamos que sea un número válido
                        if valor == '-' or pd.isna(valor):
                            continue
                            
                        try:
                            amount = float(valor)
                            if amount > 0:
                                p_id = self.map_p[p_encontrado]
                                self.Group_Inv_Data[(g_name, p_id)] = amount
                        except ValueError:
                            continue
                        # diccionario: {('A1', ID_Estabilizacion): 60.0}).

        print("Datos procesados")
    
    # Se reciben dos parametros con valor por defecto: n escenarios, desviacion estandar.
    def Generar_Escenarios_SAA(self, num_escenarios=100, variabilidad=0.20):
        
        #Genera escenarios estadísticos usando Sample Average Approximation (SAA).
        #Asume una distribución Normal (Campana de Gauss).
        
        # Guarda el valor 50
        self.N_Scenarios = num_escenarios
        # Cada escenario tiene exactamente la misma probabilidad de ocurrir (1/N)
        # Crea un Vector de Probabilidad, genera un diccionario del 1 al 50, y c/escenario
        # tiene la misma prob. de ocurrencia 1/50 = 0.02.
        # En metodo SAA, todos los escenarios son equiprobables.
        self.Escenarios = {s: 1.0 / num_escenarios for s in range(1, num_escenarios + 1)}
        self.Demanda_Estocastica = {}

        # Fijamos una semilla (seed) para que los escenarios aleatorios sean reproducibles.
        # Esto es vital para que al calcular el VSS, estemos comparando lo mismo.
        # Obliga a generar exacta% la misma secuencia de numeros pseudoaleatorios cada vez que corra el modelo.
        # Obligatiorio para el analisis de sensibilidad, para comparar parametros, deben ser los mismo 50 escenarios.
        np.random.seed(42)

        # Iteramos sobre cada demanda base del Excel
        for (n_name, t), row in self.Demands.iterrows():
            # MEDIA: Asume que las demandas del excel son las medias para cada mezcla en el mes t
            val_base = float(row['D'])
            desviacion_estandar = val_base * variabilidad
            
            # Si no hay demanda, en todos los escenarios es 0
            if val_base == 0:
                for s in range(1, num_escenarios + 1):
                    self.Demanda_Estocastica[(n_name, t, s)] = 0.0
                continue

            # Generamos 'num_escenarios' muestras desde una distribución Normal
            muestras = np.random.normal(loc=val_base, scale=desviacion_estandar, size=num_escenarios)
            
            for s in range(1, num_escenarios + 1):
                # Guardamos la muestra. Usamos max(0, ...) para evitar demandas negativas imposibles.
                self.Demanda_Estocastica[(n_name, t, s)] = max(0.0, muestras[s-1])
                  
    def Problema(self):
        model = AbstractModel()

        # Set estocástico 
        model.S = Set(initialize=list(self.Escenarios.keys()))    # Conjunto de Escenarios  
            
        # Sets
        max_t = int(self.Tiempo['t'].max())
        model.T = Set(initialize=np.arange(1, max_t + 1))             # T = 0, ..., 10 max  - Conjunto de días en el horizonte de planificación
        model.A = Set(initialize=np.arange(1, self.N_Alcoholes + 1))  # A = a1,a2,a3        - Conjunto de alcoholes    
        model.N = Set(initialize=np.arange(1, self.N_Mezclas + 1))    # N = n1.n2.n3        - Conjunto de mezclas
        model.P = Set(initialize=np.arange(1, self.N_Procesos + 1))   # P = Rec,Mix,Est,Ref - Conjunto de procesos
        
        def Tp_init(model):
            return [t for t in self.Tiempo['t'] if t != 0]
        model.Tp = Set(initialize=Tp_init)                            # T' = 1,...,10       - Conjunto de días reales del plan
        
        # Pn plano
        def Pn_init_flat(model):
            lista_tuplas = []                                         # Datos Inmutables
            for n, lista_p in self.Pn_dict.items():
                for p in lista_p:
                    lista_tuplas.append((n, p))
            print(lista_tuplas)                                       # ej.(n1, mix) : [(1, 2), (1, 4), (1, 3), (2, 2), (2, 4), (3, 2), (3, 4), (3, 3)]
            return lista_tuplas
        model.Pn = Set(dimen=2, initialize=Pn_init_flat)              # Conjunto de procesos p que pasa la mezcla n 
        
        # An plano
        def An_init_flat(model):
            lista_tuplas = []
            for n, lista_a in self.An_dict.items():
                for a in lista_a:
                    lista_tuplas.append((n, a))
            print(lista_tuplas)                                       # ej. (n1,a1) : [(1, 1), (1, 2), (2, 2), (2, 3), (3, 1), (3, 3)]
            return lista_tuplas
        model.An = Set(dimen=2, initialize=An_init_flat)              # Conjunto de alcoholes a requeridos por la mezcla n 
        
        max_tau = int(self.Routes['tau'].max())
        model.U = Set(initialize=np.arange(0, max_tau))
        #model.U = Set(initialize=[0, 1])                              # Conjunto de edades de maduración rango 0-1 
        
        # Parámetros - con valor escalar
        # L, Q_min, Cv, Ci_minus, M = 30000, 10000, 20, 100, 100000
        L, Q_min, Cv, M = 30000, 10000, 20, 100000
        Ci_minus = self.Ci_minus_val
        
        # PARÁMETROS ESTOCÁSTICOS SAA
        # Las probabilidades ahora vienen del generador SAA (todas valen 1/N)
        def pi_s(model, s): 
            return self.Escenarios[s]
        model.pi = pyo.Param(model.S, rule=pi_s)
        
        def D_nts_rule(model, n, t, s):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            # Retorna el valor simulado, si no existe retorna 0.0
            return self.Demanda_Estocastica.get((n_name, t, s), 0.0)
        model.D_nts = pyo.Param(model.N, model.Tp, model.S, rule=D_nts_rule, mutable=True)
        
        # indices (n_name, p_name) y columna Cwip
        # Coste de inventario por mantener un litro de mezcla n en el proceso p.    
        def Cwip_n(model, n, p):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            p_name = self.Procesos.iloc[p-1]['p']
            return float(self.Routes.loc[(n_name, p_name), 'Cwip']) if (n_name, p_name) in self.Routes.index else 0.0
        model.Cwip_n = Param(model.Pn, rule=Cwip_n)
        
        # indices (n_name, p_name) y columna tau
        # Días necesarios para la maduración de la mezcla n en el proceso p.
        def Tau_np(model, n, p):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            p_name = self.Procesos.iloc[p-1]['p']
            return self.Routes.loc[(n_name, p_name), 'tau'] if (n_name, p_name) in self.Routes.index else 0.0
        model.Tau_np = Param(model.Pn, rule=Tau_np)
        
        # indices (n_name, a_name) y columna R
        # Proporción de alcohol a necesaria para preparar la mezcla n en el día t.
        def R_ant(model, n, a, t):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            a_name = self.Alcoholes.iloc[a-1]['a']
            return float(self.Recipes.loc[(n_name, a_name), 'R']) if (n_name, a_name) in self.Recipes.index else 0.0
        model.R_ant = Param(model.An, model.T, rule=R_ant)
        
        # Capacidad de volumen diario del proceso p.     
        def CAP_p(model, p): return self.Capacities['CAP_p'].iloc[p-1] * self.Cap_factor
        model.CAP_p = Param(model.P, rule=CAP_p)
        
        # Capacidad de mano de obra diaria del proceso.
        def CAP_labor_p(model, p): return self.Capacities['CAP_labor'].iloc[p-1]
        model.CAP_labor_p = Param(model.P, rule=CAP_labor_p)
        
        # Parámetros Iniciales
        
        # LLegadas programadas de alcohol w_a 
        def init_Wa(model, a): 
            a_name = self.Alcoholes.iloc[a-1]['a']
            if not self.Init_Omega.empty and a_name in self.Init_Omega.index:
                return self.Init_Omega.loc[a_name, 'wa']
            return 0.0
        model.W_a = Param(model.A, initialize=init_Wa, mutable=True)      # Cantidad de alcohol a que se recibe al inicio del horizonte deplanificación.                  
        
        # Inventario Alcohol Inicial
        def init_Ia(model, a):
            a_name = self.Alcoholes.iloc[a-1]['a']
            if not self.Init_Alcoholes.empty and a_name in self.Init_Alcoholes.index:
                return self.Init_Alcoholes.loc[a_name, 'Ia']
            return 0.0
        model.I_a = Param(model.A, initialize=init_Ia, mutable=True)        # Inventario de alcohol a al inicio del horizonte de planificación. 
       
        # Backlog Inicial 
        def init_Iminus_n(model, n):
            n_name = self.Mezclas.iloc[n-1]['n']
            if not self.Init_Backlog.empty and n_name in self.Init_Backlog.index:
                return float(self.Init_Backlog.loc[n_name, 'I_minus_n'])
            return 0.0
        model.I_minus_n = Param(model.N, initialize=init_Iminus_n, mutable=True)  # Atraso de la mezcla n al inicio del horizonte de planifiación.  
        
        # Inventario con Antiguedad - gamma
        def init_gamma(model, n, p, u):
            n_name = self.Mezclas.iloc[n-1]['n']
            p_name = self.Procesos.iloc[p-1]['p']
            if not self.Init_Gamma.empty and (n_name, p_name, u) in self.Init_Gamma.index:
                return self.Init_Gamma.loc[(n_name, p_name, u), 'gamma']
            return 0.0
        model.gamma = Param(model.Pn, model.U, initialize=init_gamma, mutable=True) # Inventario de la mezcla n que ha madurado por u días en el proceso p al incio del horizonte de planificación.     
       
        # Inventario Inicial de Grupos Indiferenciados - I_gp 
        # Set auxiliar de Grupos (solo los que tienen inventario para iterar)
        # Solo crea este conjunto para las combinaciones exactas de (Grupo, Proceso) 
        # que estan en el Excel y que sí tienen inventario.
        
        model.Set_Grupos_P = Set(dimen=2, initialize=list(self.Group_Inv_Data.keys()))
        
        def init_I_Gp(model, g_name, p_id):
            return self.Group_Inv_Data.get((g_name, p_id), 0.0)
        # si por alguna razón Pyomo pregunta por una combinación que no existe en el diccionario, 
        # la función no arrojará un error que detenga el programa, sino que devolverá un 0.0
        model.I_Gp = Param(model.Set_Grupos_P, initialize=init_I_Gp, mutable=True)  # Inventario del grupo G en N(p) al finalizar el proceso p al inicio del horizonte de planificación.    

        
        # Vale 1 si el proceso no es el último, y 0 si sí lo es. Corta el flujo luego del refinado.
        def alpha_p(model, p):
            p_name = self.Procesos.iloc[p-1]['p']
            # Si el proceso es "refinement", alpha = 0 , salida
            if p_name == self.name_ref:
                return 0
            else:
                return 1
        model.alpha_p = Param(model.P, rule=alpha_p)
        
        # Topología 
        def get_next_process(n_id, p_id):
            p_curr_name = self.Procesos.iloc[p_id-1]['p']
            n_curr_name = self.Mezclas.iloc[n_id-1]['n']
            
            # 1. Si el proceso actual es 'mixing'
            if p_curr_name == self.name_mix:
                if n_curr_name == 'n2': 
                    return self.map_p[self.name_ref]
                elif n_curr_name == 'n3':
                    return self.map_p['barrel']
                elif n_curr_name in ['n4', 'n5', 'n8']:
                    return self.map_p['infusion']
                else:
                    # Para n1, n6, n7
                    return self.map_p['stabilization']
                    
            # 2. Si el proceso actual es 'barrel aging' (solo aplica a n3)
            if p_curr_name == 'barrel':
                return self.map_p['stabilization']
                
            # 3. Si el proceso actual es 'w. infusion' (aplica a n4, n5, n8)
            if p_curr_name == 'infusion':
                return self.map_p['stabilization']
            
            # 4. Si el proceso actual es 'stabilization' (aplica a todos menos n2)
            if p_curr_name == 'stabilization':
                return self.map_p[self.name_ref]
            
            # Si el proceso es 'refinement' (el último) o no coincide con nada
            return None
        
        
        # Variables
        model.q = Var(model.A, model.Tp, within=NonNegativeReals, initialize=0)                     # 𝐶𝑎𝑛𝑡𝑖𝑑𝑎𝑑 𝑑𝑒 𝑎𝑙𝑐𝑜ℎ𝑜𝑙 𝛼 𝑟𝑒𝑐𝑖𝑏𝑖𝑑𝑜 𝑒𝑛 𝑒𝑙 𝑑í𝑎 𝑡. 
        model.l = Var(model.A, model.Tp, within=NonNegativeIntegers, initialize=0)                  # 𝐿𝑜𝑡𝑒𝑠 𝑑𝑒 𝑎𝑙𝑐𝑜ℎ𝑜𝑙 𝛼 𝑟𝑒𝑐𝑖𝑏𝑖𝑑𝑜 𝑒𝑛 𝑒𝑙 𝑑í𝑎 𝑡.       
        model.v = Var(model.A, model.T, within=NonNegativeReals, initialize=0)                      # 𝑆𝑡𝑜𝑐𝑘 𝑑𝑒 𝑎𝑙𝑐𝑜ℎ𝑜𝑙 𝛼 𝑞𝑢𝑒 𝑒𝑠𝑡á 𝑙𝑖𝑠𝑡𝑜 𝑒𝑛 𝑒𝑙 𝑑í𝑎 𝑡.    
        model.w = Var(model.An, model.Tp, within=NonNegativeReals, initialize=0)                    # 𝐶𝑎𝑛𝑡𝑖𝑑𝑎𝑑 𝑑𝑒 𝑎𝑙𝑐𝑜ℎ𝑜𝑙 𝛼 𝑚𝑒𝑧𝑐𝑙𝑎𝑑𝑜 𝑝𝑎𝑟𝑎 𝑝𝑟𝑒𝑝𝑎𝑟𝑎𝑟 𝑙𝑎 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑒𝑛 𝑒𝑙 𝑑í𝑎 𝑡. 
        model.y = Var(model.N, model.Tp, within=NonNegativeReals, initialize=0)                     # 𝐶𝑎𝑛𝑡𝑖𝑑𝑎𝑑 𝑑𝑒 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑝𝑟𝑒𝑝𝑎𝑟𝑎𝑑𝑎 𝑒𝑛 𝑒𝑙 𝑑í𝑎 𝑡. 
        model.z = Var(model.Pn, model.T, model.U, within=NonNegativeReals, initialize=0)            # 𝐶𝑎𝑛𝑡𝑖𝑑𝑎𝑑 𝑑𝑒 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑞𝑢𝑒 ℎ𝑎 𝑚𝑎𝑑𝑢𝑟𝑎𝑑𝑜 𝑑𝑢𝑟𝑎𝑛𝑡𝑒 𝑢 𝑑í𝑎𝑠 𝑒n e𝑙 𝑝𝑟𝑜𝑐𝑒𝑠𝑜 𝑝 𝑎𝑙 𝑓𝑖𝑛𝑎𝑙 𝑑𝑒𝑙 𝑑í𝑎 𝑡.
        model.k = Var(model.Pn, model.T, model.S, within=NonNegativeReals, initialize=0)            # 𝑆𝑡𝑜𝑐𝑘 𝑑𝑒 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑞𝑢𝑒 𝑒𝑠𝑡á 𝑙𝑖𝑠𝑡𝑎 𝑒𝑛 𝑒𝑙 𝑝𝑟𝑜𝑐𝑒𝑠𝑜 𝑝 𝑎𝑙 𝑓𝑖𝑛𝑎𝑙 𝑑𝑒𝑙  𝑑í𝑎 𝑡.      
        model.x = Var(model.N, model.P, model.P, model.Tp, within=NonNegativeReals, initialize=0)   # 𝐶𝑎𝑛𝑡𝑖𝑑𝑎𝑑 𝑑𝑒 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑙𝑖𝑠𝑡𝑎 𝑒𝑛 𝑒𝑙 𝑝𝑟𝑜𝑐𝑒𝑠𝑜 𝑝 𝑞𝑢𝑒 𝑝𝑎𝑠𝑎 𝑎𝑙 𝑝𝑟𝑜𝑐𝑒𝑠𝑜 𝑑 e𝑛 𝑒𝑙 𝑑í𝑎 𝑡. 
        model.i_minus = Var(model.N, model.T, model.S, within=NonNegativeReals, initialize=0)       # 𝐴𝑡𝑟𝑎𝑠𝑜 𝑑𝑒 𝑙𝑎 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑎𝑙 𝑓𝑖𝑛𝑎𝑙 𝑑𝑒𝑙 𝑑í𝑎 𝑡. 
        model.b = Var(model.Pn, within=NonNegativeReals, initialize=0)                              # 𝐼𝑛𝑣𝑒𝑛𝑡𝑎𝑟𝑖𝑜 𝑑𝑒 𝑙𝑎 𝑚𝑒𝑧𝑐𝑙𝑎 𝑛 𝑒𝑛 𝑒𝑙 𝑝𝑟𝑜𝑐𝑒𝑠𝑜 𝑝 𝑎𝑙 𝑖𝑛𝑖𝑐𝑖𝑜 𝑑𝑒𝑙 ℎ𝑜𝑟𝑖𝑧𝑜𝑛𝑡e d𝑒 𝑝𝑙𝑎𝑛𝑖𝑓𝑖𝑐𝑎𝑐𝑖ón.
        model.delta = Var(model.Pn, model.Tp, within=Binary, initialize=0)                          # 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 𝑏𝑖𝑛𝑎𝑟𝑖𝑎 𝑞𝑢𝑒 𝑡𝑜𝑚𝑎 𝑒𝑙 𝑣𝑎𝑙𝑜𝑟 1 𝑠𝑖 𝒛𝒏𝒖𝒑𝒕 > 0, y 0 𝑒𝑛 𝑐𝑎𝑠𝑜 𝑐𝑜𝑛𝑡𝑟𝑎𝑟𝑖𝑜.
        
        # Holgura de capacidad (Litros extra que desbordan el estanque)
        #model.slack_cap = Var(model.P, model.Tp, model.S, within=NonNegativeReals, initialize=0)
        
        # Función Objetivo
        # Coste I: Inventario de alcohol crudo (Primera Etapa)
        def exp_coste_I(model):
            return sum(Cv * model.v[a,t] for a in model.A for t in model.Tp)
        model.coste_I = Expression(rule=exp_coste_I)
        # Coste II: Inventario en proceso / maduración (Primera Etapa)
        def exp_coste_II(model):
            return sum(model.Cwip_n[n,p] * model.z[n,p,t,u] 
                       for (n,p) in model.Pn for t in model.Tp for u in model.U 
                       if u < value(model.Tau_np[n, p]))
        model.coste_II = Expression(rule=exp_coste_II)
        # Coste III: Valor Esperado del inventario de producto terminado (Segunda Etapa)
        def exp_coste_III(model):
            return sum(model.pi[s] * model.Cwip_n[n,p] * model.k[n,p,t,s] 
                       for s in model.S for (n,p) in model.Pn for t in model.Tp)
        model.coste_III = Expression(rule=exp_coste_III)
        # Coste IV: Valor Esperado del backlog / atraso (Segunda Etapa)
        def exp_coste_IV(model):
            return sum(model.pi[s] * Ci_minus * model.i_minus[n,t,s] 
                       for s in model.S for n in model.N for t in model.Tp)
        model.coste_IV = Expression(rule=exp_coste_IV)  
        
        # Costo V: Penalización por rebasar la capacidad física (bodegaje externo)
        # penalizacion = 1000 
        # def exp_coste_V(model):
        #     return sum(model.pi[s] * penalizacion * model.slack_cap[p,t,s] 
        #                for p in model.P for t in model.Tp for s in model.S)
        # model.coste_V = Expression(rule=exp_coste_V)

        # Función Objetivo Estocástica
        def obj_rule(model):
            return model.coste_I + model.coste_II + model.coste_III + model.coste_IV
        model.FO = Objective(rule=obj_rule, sense=minimize)
        
        # Restricciones
        
        def suma_prob(model):
            suma = sum(value(model.pi[s]) for s in model.S)    
            # round(..., 4) evita error de precisión decimal y se verifica que suman 1
            if round(suma, 4) == 1.0:
                return Constraint.Skip  # no se manda al solver
            else:
                raise ValueError(f"Error en probabilidades: {suma}")
        model.const_suma_prob = Constraint(rule=suma_prob)
            
        # (1) Balance Producto Terminado (k_npt) 
        # El líquido entró durante nuestro plan actual. t > tau_val
        # Si hoy es el Día 5 y la maduración dura 2 días, la entrada de hoy es el líquido que tenía edad 0 hace 2 días
        def balance_k(model, n, p, t, s):
            k_ayer = model.k[n, p, t-1, s] if t > 1 else model.b[n,p]           # restriccion 4
            tau_val = int(value(model.Tau_np[n,p])) #maduración
            entrada = 0
            if t > tau_val:
                entrada = model.z[n, p, t-tau_val, 0]
            else:
                # Esto cubre t == tau_val (u=0) y t < tau_val (u>0)
                # La edad 'u' que tenía en el pasado y que madura hoy es (tau_val - t)
                entrada = model.gamma[n, p, tau_val - t]                        # restriccion 7                                  # restriccion 7
            
            salida = 0
            
            #recuperamos el valor de alpha para este proceso
            p_intermedio = (model.alpha_p[p] == 1)
            
            if p_intermedio:
                p_next = get_next_process(n, p)
                if p_next is not None:
                    salida = model.x[n, p, p_next, t]
            else: 
                #Es Refinado (alpha=0): Sale a demanda estocástica
                backlog_ayer = model.i_minus[n, t-1,s] if t > 1 else model.I_minus_n[n]       # restriccion 8
                salida = model.D_nts[n,t,s] + backlog_ayer - model.i_minus[n,t,s]

            return model.k[n,p,t, s] == k_ayer + entrada - salida
        model.const_bal_k = Constraint(model.Pn, model.Tp, model.S, rule=balance_k)
        
        # (2) Balance Alcohol
        def balance_alcohol(model, a, t):
            v_ayer = model.v[a, t-1] if t > 1 else model.I_a[a]                 # restriccion 5
            q_ayer = model.q[a, t-1] if t > 1 else model.W_a[a]               # restriccion 6
            consumo = sum(model.w[n, a, t] for n in model.N if (n,a) in model.An)
            return model.v[a,t] == v_ayer + q_ayer - consumo
        model.const_bal_alc = Constraint(model.A, model.Tp, rule=balance_alcohol)
        
        # (3) Consistencia de inventario inicial b_np  para Grupos Indiferenciados, inventario real inicial  I_Gp
        def consistencia_inv_grupos(model, g_name, p_id):
            # Recuperamos la lista de IDs de mezcla para este grupo
            lista_mezclas_ids = self.Group_Mapping.get((g_name, p_id), [])
            if not lista_mezclas_ids:
                return Constraint.Skip
            
            # Sumamos las variables 'b' de esas mezclas
            suma_b = sum(model.b[n_id, p_id] for n_id in lista_mezclas_ids)
            
            return suma_b == model.I_Gp[g_name, p_id]
        model.const_cons_inv_g = Constraint(model.Set_Grupos_P, rule=consistencia_inv_grupos)
        
        # 4, 5, 6, 7, 8
        
        # (4) El inventario listo al inicio 𝑘𝑛𝑝0 (t = 0) es igual al dato histórico 𝑏𝑛𝑝
        # def inventario_listo_inicial(model, n, p): return model.k[n, p, 0] == model.b[n,p]
        # model.const_inv_listo_inicial = Constraint(model.Pn, rule=inventario_listo_inicial)

        # (9) Capacidad de Almacenamiento Volumétrica
        def capacidad_volumen(model, p, t, s):
            p_recep = self.map_p[self.name_recep]
            if p == p_recep: 
                return Constraint.Skip
            volumen_wip = sum(sum(model.z[n, p, t, u] for u in model.U if u < value(model.Tau_np[n, p])) 
                              for n in model.N if (n, p) in model.Pn)
            volumen_k = sum(model.k[n, p, t, s] for n in model.N if (n, p) in model.Pn)
            #+HOLGURA
            return volumen_wip + volumen_k <= model.CAP_p[p]
        model.const_cap_vol = Constraint(model.P, model.Tp, model.S, rule=capacidad_volumen)
        
        # (10) Capacidad de Mano de Obra (Flujo Diario)
        def capacidad_mano_obra(model, p, t):
            p_recep = self.map_p[self.name_recep]
            if p == p_recep: 
                return Constraint.Skip
            flujo_diario = sum(model.z[n, p, t, 0] for n in model.N if (n, p) in model.Pn)
            return flujo_diario <= model.CAP_labor_p[p]
        model.const_cap_man = Constraint(model.P, model.Tp, rule=capacidad_mano_obra)
        
        # (11) Capacidad de Recepción de Alcoholes
        def capacidad_recepcion(model, t):
            p_recep = self.map_p[self.name_recep]
            carga_total = sum(model.q[a, t] + model.v[a, t] for a in model.A)
            return carga_total <= model.CAP_p[p_recep]
        model.const_cap_recep = Constraint(model.Tp, rule=capacidad_recepcion)
        
        # (12) Capacidad de Mano de Obra en Recepción
        def capacidad_recepcion_man(model, t):
            p_recep = self.map_p[self.name_recep]
            descarga_total = sum(model.q[a, t] for a in model.A)
            return descarga_total <= model.CAP_labor_p[p_recep]
        model.const_cap_recep_man = Constraint(model.Tp, rule=capacidad_recepcion_man)
        
        # (13) Tamaño de Lote Mínimo
        def min_lot_size(model, n, p, t): 
            return Q_min * model.delta[n, p, t] <= model.z[n, p, t, 0]
        model.const_min_lot = Constraint(model.Pn, model.Tp, rule=min_lot_size)
        
        # (14) Cota Superior (Big M), limitado por su capacidad fisica
        def max_lot_size(model, n, p, t): 
            return model.z[n, p, t, 0] <= M * model.delta[n, p, t]
        model.const_max_lot = Constraint(model.Pn, model.Tp, rule=max_lot_size)
        
        # (15) Lotes de Camiones (Recepción de Alcohol en múltiplos de L)
        def trucks(model, a, t): 
            return model.q[a,t] == L * model.l[a,t]
        model.const_trucks = Constraint(model.A, model.Tp, rule=trucks)

        # (16) Dinámica de Envejecimiento (Z_u -> Z_u+1)
        def envejecimiento(model, n, p, t, u):
            tau_lim = int(value(model.Tau_np[n, p]))
            # la mezcla pasa por las edades 0 y 1, en 2 pasa de z a k
            if u >= tau_lim - 1: 
                return Constraint.Skip
            if t == 1: 
                return model.z[n, p, t, u + 1] == model.gamma[n, p, u]          # restriccion 7 para -t
            else: 
                return model.z[n, p, t, u + 1] == model.z[n, p, t-1, u]
        model.const_envej = Constraint(model.Pn, model.Tp, model.U, rule=envejecimiento)

        # (17) Continuidad de Flujo (Salida x es Entrada z_edad_0)
        def flow_continuity(model, n, p, t):
            p_next = get_next_process(n, p)
            if p_next is None:                          #refinado
                return Constraint.Skip
            if (n, p_next) not in model.Pn:             # existe n para p_next
                return Constraint.Skip
            return model.z[n, p_next, t, 0] == model.x[n, p, p_next, t] 
        model.const_flow_continuity = Constraint(model.Pn, model.Tp, rule=flow_continuity)
        
        # (18) Entrada a Proceso de Mezcla (y_nt a z_edad_0)
        def entrada_mix(model, n, t):
            p_mix = self.map_p[self.name_mix]
            if (n, p_mix) not in model.Pn:              # existe n para mix
                return Constraint.Skip
            return model.z[n, p_mix, t, 0] == model.y[n, t]
        model.const_ent_mix = Constraint(model.N, model.Tp, rule=entrada_mix)
        
        # (19) Receta de Mezcla 
        def receta(model, n, a, t): 
            return model.y[n, t] * model.R_ant[n, a, t] == model.w[n, a, t]
        model.const_receta = Constraint(model.An, model.Tp, rule=receta)

        return model.create_instance()
    
    def ExportarResultados(self, instance, nombre_archivo="Resultados_Estocásticos.xlsx"):
        
        # ID a NOMBRE
        # Pyomo usa índices base 1 (1,2,3), Pandas base 0. Por eso el [id-1].
        def get_n(n_id): return self.Mezclas.iloc[n_id-1]['n']
        def get_a(a_id): return self.Alcoholes.iloc[a_id-1]['a']
        def get_p(p_id): return self.Procesos.iloc[p_id-1]['p']
        
        # Variable y_nt (Producción Mezcla)
        df_y = pd.DataFrame([
            (get_n(n), t, round(value(instance.y[n, t]), 2)) # Aquí cambia n por get_n(n)
            for n in instance.N for t in instance.Tp 
            if value(instance.y[n, t]) > 0.001
        ], columns=["Mezcla", "Dia", "Litros_Producidos - y_nt"])

        # Variable v_at (Inventario Alcohol)
        # Usa instance.T o instance.Tp dependiendo de si t=0
        times = instance.T if hasattr(instance, 'T') else instance.Tp
        df_v = pd.DataFrame([
            (get_a(a), t, round(value(instance.v[a, t]), 2)) # a por get_a(a)
            for a in instance.A for t in times
            if t <= self.N_Tiempo 
        ], columns=["Alcohol", "Dia", "Stock_Tanque - v_at"])

        # Variable k_npt (Stock Listo)
        df_k = pd.DataFrame([
            (get_n(n), get_p(p), t, s, round(value(instance.k[n, p, t, s]), 2)) # n, p traducidos
            for (n,p) in instance.Pn for t in instance.Tp for s in instance.S # Tp (días operativos)
            if value(instance.k[n, p, t, s]) > 0.001
        ], columns=["Mezcla", "Proceso", "Dia", "Escenario", "Stock_Listo_k - k_npt"])
        
        # Variable i_minus_nt (Backlog)
        df_i = pd.DataFrame([
            (get_n(n), t, s, round(value(instance.i_minus[n, t, s]), 2)) #  n traducido
            for n in instance.N for t in instance.Tp for s in instance.S
            if value(instance.i_minus[n, t, s]) > 0.001
        ], columns=["Mezcla", "Dia", "Escenario", "Backlog - i_minus_nt"])

        # Variable z_nupt (Detalle Envejecimiento)
        df_z = pd.DataFrame([
            (get_n(n), get_p(p), t, u, round(value(instance.z[n, p, t, u]), 2)) # n, p traducidos
            for (n,p) in instance.Pn for t in instance.Tp for u in instance.U
            if value(instance.z[n, p, t, u]) > 0.001
        ], columns=["Mezcla", "Proceso", "Dia", "Edad", "Stock_Proceso_z - z_nupt"])

        # Variable q_at (Recepcion)
        df_q = pd.DataFrame([
            (get_a(a), t, round(value(instance.q[a, t]), 2)) #  a traducido
            for a in instance.A for t in instance.Tp
            if value(instance.q[a, t]) > 0.001
        ], columns=["Alcohol", "Dia", "Recibido - q_at"])

        # Guardar en archivo Excel
        try:
            with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
                # Escribimos, y si están vacíos creamos una hoja dummy para que no falle
                if not df_y.empty: df_y.to_excel(writer, sheet_name="y_produccion", index=False)
                else: pd.DataFrame(columns=df_y.columns).to_excel(writer, sheet_name="y_produccion", index=False)

                if not df_v.empty: df_v.to_excel(writer, sheet_name="v_inv_alcohol", index=False)
                else: pd.DataFrame(columns=df_v.columns).to_excel(writer, sheet_name="v_inv_alcohol", index=False)

                if not df_k.empty: df_k.to_excel(writer, sheet_name="k_stock_listo", index=False)
                else: pd.DataFrame(columns=df_k.columns).to_excel(writer, sheet_name="k_stock_listo", index=False)

                if not df_i.empty: df_i.to_excel(writer, sheet_name="i_backlog", index=False)
                else: pd.DataFrame(columns=df_i.columns).to_excel(writer, sheet_name="i_backlog", index=False)

                if not df_q.empty: df_q.to_excel(writer, sheet_name="q_recepcion", index=False)
                else: pd.DataFrame(columns=df_q.columns).to_excel(writer, sheet_name="q_recepcion", index=False)

                if not df_z.empty: df_z.to_excel(writer, sheet_name="z_en_proceso", index=False)
                else: pd.DataFrame(columns=df_z.columns).to_excel(writer, sheet_name="z_en_proceso", index=False)
                
            print(f"Resultados exportados correctamente a {nombre_archivo}")
            
        except PermissionError:
            print(f"Error: El archivo '{nombre_archivo}' está abierto en Excel.")

    def Solver(self):
        instance = self.Problema()
        solver = SolverFactory('gurobi')
        solver.options['TimeLimit'] = 240
        solver.options['MIPGap'] = 0.001 
        results = solver.solve(instance, tee=True)
        
        if (results.solver.status == pyo.SolverStatus.ok) and \
           (results.solver.termination_condition == pyo.TerminationCondition.optimal):
            # --- REPORTE DE COSTOS EN CONSOLA ---
            print('\n' + '='*50)
            print('SOLUCIÓN ÓPTIMA ENCONTRADA')
            print('='*50)
            print('COSTOS DE PRIMERA ETAPA (Decisiones Fijas):')
            print(f'  - Costo I (Almacenamiento Alcohol):   $ {value(instance.coste_I):.2f}')
            print(f'  - Costo II (Inventario en Maduración):$ {value(instance.coste_II):.2f}')
            print('\nCOSTOS DE SEGUNDA ETAPA (Valor Esperado):')
            print(f'  - Costo III (Stock Terminado):        $ {value(instance.coste_III):.2f}')
            print(f'  - Costo IV (Penalización Backlog):    $ {value(instance.coste_IV):.2f}')
            print('-'*50)
            print(f'COSTO TOTAL DE LA FUNCIÓN OBJETIVO:  $ {value(instance.FO):.2f}')
            print('='*50 + '\n')
            
            # EL COSTO 2 ES LA INVERSIO Y EL COSTO 4 EL RIESGO.

            self.ExportarResultados(instance)
        else:
            print("Error: No se encontró solución óptima.")
        return instance

if __name__ == "__main__":
    modelo = PiscoModel()
    modelo.ReadExcelFile(r"/Users/diegopazdelavega/Documents/Proyecto/capel/Codigos/Datos_Originales.xlsx")
    modelo.Generar_Escenarios_SAA(num_escenarios=100, variabilidad=0.20)
    instancia = modelo.Solver()
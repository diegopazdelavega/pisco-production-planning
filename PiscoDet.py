import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Set, Param, Var, Constraint, Objective, 
    Binary, NonNegativeReals, NonNegativeIntegers, minimize, value,
    Expression
)
from solver_utils import GurobiConfigurationError, create_gurobi_solver

class PiscoModel_Operativo: 
    def __init__(self, name=None):
        self.data = {}
        
    def ReadExcelFile(self, FileName):
        try:
            self.data = pd.read_excel(FileName, sheet_name=None)
        except FileNotFoundError:
            return

        for sheet, df in self.data.items():
            df.columns = df.columns.str.strip()

        self.Tiempo = self.data['Set_T']
        self.Alcoholes = self.data['Set_A']
        self.Mezclas = self.data['Set_N']
        self.Procesos = self.data['Set_P']
        self.Scalars = self.data['Scalars']
        self.Routes = self.data['Routes_Taus']  
        self.Recipes = self.data['Recipes']     
        self.Demands = self.data['Demands']     
        self.Capacities = self.data['Capacities'].set_index('p')  
        
        self.N_Tiempo = len(self.Tiempo)
        self.N_Alcoholes = len(self.Alcoholes)
        self.N_Mezclas = len(self.Mezclas)
        self.N_Procesos = len(self.Procesos)

        self.map_n = {nombre: i+1 for i, nombre in enumerate(self.Mezclas['n'])}
        self.map_p = {nombre: i+1 for i, nombre in enumerate(self.Procesos['p'])}
        self.map_a = {nombre: i+1 for i, nombre in enumerate(self.Alcoholes['a'])}

        self.Pn_dict = {i+1: [] for i in range(len(self.Mezclas))}
        for index, row in self.Routes.iterrows():
            if row['n'] in self.map_n and row['p'] in self.map_p:
                if row['p'] == 'reception':                                     
                    continue
                self.Pn_dict[self.map_n[row['n']]].append(self.map_p[row['p']])
        
        self.An_dict = {i+1: [] for i in range(len(self.Mezclas))}
        for index, row in self.Recipes.iterrows():
            if row['n'] in self.map_n and row['a'] in self.map_a:
                self.An_dict[self.map_n[row['n']]].append(self.map_a[row['a']])

        self.Routes = self.Routes.set_index(['n', 'p'])
        self.Recipes = self.Recipes.set_index(['n', 'a'])
        self.Demands = self.Demands.set_index(['n', 't'])

        self.name_recep = 'reception'
        self.name_mix = 'mixing'
        self.name_ref = 'refinement'
        
        self.Init_Alcoholes = self.data['Inv_Inicial_Alcoholes'].set_index('a')  
        self.Init_Omega = self.data['LLegadas_Alcoholes_Omega'].set_index('a')     
        self.Init_Backlog = self.data['Backlog_Inicial'].set_index('n')          
        self.Init_Gamma = self.data['Inventario_Antiguo_Gamma'].set_index(['n','p','u']) 
        
        self.Group_Mapping = {} 
        self.Group_Inv_Data = {}

        if 'Asignacion_de_Grupos_N(p)' in self.data:
            df_map = self.data['Asignacion_de_Grupos_N(p)']
            for index, row in df_map.iterrows():
                g_name = row['Group'] 
                for col_name in df_map.columns:
                    if col_name == 'Group': continue 
                    p_encontrado = None
                    for p_real in self.map_p.keys():
                        if p_real in col_name: 
                            p_encontrado = p_real
                            break
                    if p_encontrado:
                        celda = str(row[col_name]) 
                        if celda == '-' or pd.isna(row[col_name]) or celda.strip() == '':
                            continue
                        mezclas_en_celda = [m.strip() for m in celda.split(',')]
                        p_id = self.map_p[p_encontrado]
                        key = (g_name, p_id)
                        if key not in self.Group_Mapping:
                            self.Group_Mapping[key] = []
                        for m_nombre in mezclas_en_celda:
                            if m_nombre in self.map_n:
                                self.Group_Mapping[key].append(self.map_n[m_nombre])

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
                        if valor == '-' or pd.isna(valor):
                            continue
                        try:
                            amount = float(valor)
                            if amount > 0:
                                p_id = self.map_p[p_encontrado]
                                self.Group_Inv_Data[(g_name, p_id)] = amount
                        except ValueError:
                            continue

        print("Datos procesados")

    def Problema(self):
        model = AbstractModel()

        max_t = int(self.Tiempo['t'].max())
        model.T = Set(initialize=np.arange(1, max_t + 1))             
        model.A = Set(initialize=np.arange(1, self.N_Alcoholes + 1))  
        model.N = Set(initialize=np.arange(1, self.N_Mezclas + 1))    
        model.P = Set(initialize=np.arange(1, self.N_Procesos + 1))   
        
        def Tp_init(model):
            return [t for t in self.Tiempo['t'] if t != 0]
        model.Tp = Set(initialize=Tp_init)                            
        
        def Pn_init_flat(model):
            lista_tuplas = []                                         
            for n, lista_p in self.Pn_dict.items():
                for p in lista_p:
                    lista_tuplas.append((n, p))
            return lista_tuplas
        model.Pn = Set(dimen=2, initialize=Pn_init_flat)              
        
        def An_init_flat(model):
            lista_tuplas = []
            for n, lista_a in self.An_dict.items():
                for a in lista_a:
                    lista_tuplas.append((n, a))
            return lista_tuplas
        model.An = Set(dimen=2, initialize=An_init_flat)              
        
        max_tau = int(self.Routes['tau'].max())
        model.U = Set(initialize=np.arange(0, max_tau))
        
        # L, Q_min, Cv, Ci_minus, M = 30, 20, 1, 150, 10000
        
        L, Q_min, Cv, Ci_minus, M = 30000, 10000, 20, 100, 100000        
        
        def D_nt(model, n, t):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            if (n_name, t) in self.Demands.index:
                return float(self.Demands.loc[(n_name, t), 'D'])
            return 0.0
        model.D_nt = Param(model.N, model.Tp, rule=D_nt, mutable=True)
        
        def Cwip_n(model, n, p):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            p_name = self.Procesos.iloc[p-1]['p']
            return float(self.Routes.loc[(n_name, p_name), 'Cwip']) if (n_name, p_name) in self.Routes.index else 0.0
        model.Cwip_n = Param(model.Pn, rule=Cwip_n)
        
        def Tau_np(model, n, p):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            p_name = self.Procesos.iloc[p-1]['p']
            return self.Routes.loc[(n_name, p_name), 'tau'] if (n_name, p_name) in self.Routes.index else 0.0
        model.Tau_np = Param(model.Pn, rule=Tau_np)
        
        def R_ant(model, n, a, t):
            n_name = self.Mezclas.iloc[n-1]['n'] 
            a_name = self.Alcoholes.iloc[a-1]['a']
            return float(self.Recipes.loc[(n_name, a_name), 'R']) if (n_name, a_name) in self.Recipes.index else 0.0
        model.R_ant = Param(model.An, model.T, rule=R_ant)
        
        def CAP_p(model, p): return self.Capacities['CAP_p'].iloc[p-1]
        model.CAP_p = Param(model.P, rule=CAP_p)
        
        def CAP_labor_p(model, p): return self.Capacities['CAP_labor'].iloc[p-1]
        model.CAP_labor_p = Param(model.P, rule=CAP_labor_p)
        
        def init_Wa(model, a): 
            a_name = self.Alcoholes.iloc[a-1]['a']
            if not self.Init_Omega.empty and a_name in self.Init_Omega.index:
                return self.Init_Omega.loc[a_name, 'wa']
            return 0.0
        model.W_a = Param(model.A, initialize=init_Wa, mutable=True)      
        
        def init_Ia(model, a):
            a_name = self.Alcoholes.iloc[a-1]['a']
            if not self.Init_Alcoholes.empty and a_name in self.Init_Alcoholes.index:
                return self.Init_Alcoholes.loc[a_name, 'Ia']
            return 0.0
        model.I_a = Param(model.A, initialize=init_Ia, mutable=True)        
       
        def init_Iminus_n(model, n):
            n_name = self.Mezclas.iloc[n-1]['n']
            if not self.Init_Backlog.empty and n_name in self.Init_Backlog.index:
                return float(self.Init_Backlog.loc[n_name, 'I_minus_n'])
            return 0.0
        model.I_minus_n = Param(model.N, initialize=init_Iminus_n, mutable=True)  
        
        def init_gamma(model, n, p, u):
            n_name = self.Mezclas.iloc[n-1]['n']
            p_name = self.Procesos.iloc[p-1]['p']
            if not self.Init_Gamma.empty and (n_name, p_name, u) in self.Init_Gamma.index:
                return self.Init_Gamma.loc[(n_name, p_name, u), 'gamma']
            return 0.0
        model.gamma = Param(model.Pn, model.U, initialize=init_gamma, mutable=True) 
       
        model.Set_Grupos_P = Set(dimen=2, initialize=list(self.Group_Inv_Data.keys()))
        def init_I_Gp(model, g_name, p_id):
            return self.Group_Inv_Data.get((g_name, p_id), 0.0)
        model.I_Gp = Param(model.Set_Grupos_P, initialize=init_I_Gp, mutable=True)  
        
        def alpha_p(model, p):
            p_name = self.Procesos.iloc[p-1]['p']
            if p_name == self.name_ref:
                return 0
            else:
                return 1
        model.alpha_p = Param(model.P, rule=alpha_p)
        
        #Topologia Capel
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
        model.q = Var(model.A, model.Tp, within=NonNegativeReals, initialize=0)                     
        model.l = Var(model.A, model.Tp, within=NonNegativeIntegers, initialize=0)                  
        model.v = Var(model.A, model.T, within=NonNegativeReals, initialize=0)                      
        model.w = Var(model.An, model.Tp, within=NonNegativeReals, initialize=0)                    
        model.y = Var(model.N, model.Tp, within=NonNegativeReals, initialize=0)                     
        model.z = Var(model.Pn, model.T, model.U, within=NonNegativeReals, initialize=0)            
        model.k = Var(model.Pn, model.T, within=NonNegativeReals, initialize=0)             
        model.x = Var(model.N, model.P, model.P, model.Tp, within=NonNegativeReals, initialize=0)   
        model.i_minus = Var(model.N, model.T, within=NonNegativeReals, initialize=0)        
        model.b = Var(model.Pn, within=NonNegativeReals, initialize=0)                              
        model.delta = Var(model.Pn, model.Tp, within=Binary, initialize=0)           

        # NUEVA VARIABLE DE HOLGURA DETERMINISTA
        #model.slack_cap = Var(model.P, model.Tp, within=NonNegativeReals, initialize=0)               
        
        # Función Objetivo (Puramente Determinista)
        def exp_coste_I(model):
            return sum(Cv * model.v[a,t] for a in model.A for t in model.Tp)
        model.coste_I = Expression(rule=exp_coste_I)

        def exp_coste_II(model):
            return sum(model.Cwip_n[n,p] * model.z[n,p,t,u] 
                       for (n,p) in model.Pn for t in model.Tp for u in model.U 
                       if u < value(model.Tau_np[n, p]))
        model.coste_II = Expression(rule=exp_coste_II)

        def exp_coste_III(model):
            return sum(model.Cwip_n[n,p] * model.k[n,p,t] 
                       for (n,p) in model.Pn for t in model.Tp)
        model.coste_III = Expression(rule=exp_coste_III)

        def exp_coste_IV(model):
            return sum(Ci_minus * model.i_minus[n,t] 
                       for n in model.N for t in model.Tp)
        model.coste_IV = Expression(rule=exp_coste_IV)  
        
        # COSTO V: Multa por sobrecapacidad
        # PENALIDAD_OVERFLOW = 1000
        # def exp_coste_V(model):
        #     return sum(PENALIDAD_OVERFLOW * model.slack_cap[p,t] 
        #                for p in model.P for t in model.Tp)
        # model.coste_V = Expression(rule=exp_coste_V)

        def obj_rule(model):
            return model.coste_I + model.coste_II + model.coste_III + model.coste_IV
        model.FO = Objective(rule=obj_rule, sense=minimize)
                    
        # (1) Balance Producto Terminado (k_npt) 
        def balance_k(model, n, p, t):
            k_ayer = model.k[n, p, t-1] if t > 1 else model.b[n,p]            
            tau_val = int(value(model.Tau_np[n,p])) 
            entrada = 0
            if t > tau_val:
                entrada = model.z[n, p, t-tau_val, 0]
            else:
                entrada = model.gamma[n, p, tau_val - t]                                                                               
            
            salida = 0
            p_intermedio = (model.alpha_p[p] == 1)
            
            if p_intermedio:
                p_next = get_next_process(n, p)
                if p_next is not None:
                    salida = model.x[n, p, p_next, t]
            else: 
                backlog_ayer = model.i_minus[n, t-1] if t > 1 else model.I_minus_n[n]       
                salida = model.D_nt[n,t] + backlog_ayer - model.i_minus[n,t]

            return model.k[n,p,t] == k_ayer + entrada - salida
        model.const_bal_k = Constraint(model.Pn, model.Tp, rule=balance_k)
        
        def balance_alcohol(model, a, t):
            v_ayer = model.v[a, t-1] if t > 1 else model.I_a[a]                 
            q_ayer = model.q[a, t-1] if t > 1 else model.W_a[a]               
            consumo = sum(model.w[n, a, t] for n in model.N if (n,a) in model.An)
            return model.v[a,t] == v_ayer + q_ayer - consumo
        model.const_bal_alc = Constraint(model.A, model.Tp, rule=balance_alcohol)
        
        def consistencia_inv_grupos(model, g_name, p_id):
            lista_mezclas_ids = self.Group_Mapping.get((g_name, p_id), [])
            if not lista_mezclas_ids:
                return Constraint.Skip
            suma_b = sum(model.b[n_id, p_id] for n_id in lista_mezclas_ids)
            return suma_b == model.I_Gp[g_name, p_id]
        model.const_cons_inv_g = Constraint(model.Set_Grupos_P, rule=consistencia_inv_grupos)
        
        # (9) Capacidad de Almacenamiento Volumétrica
        def capacidad_volumen(model, p, t):
            p_recep = self.map_p[self.name_recep]
            if p == p_recep: 
                return Constraint.Skip
            volumen_wip = sum(sum(model.z[n, p, t, u] for u in model.U if u < value(model.Tau_np[n, p])) 
                              for n in model.N if (n, p) in model.Pn)
            volumen_k = sum(model.k[n, p, t] for n in model.N if (n, p) in model.Pn) # Sin 's'
            return volumen_wip + volumen_k <= model.CAP_p[p]
        model.const_cap_vol = Constraint(model.P, model.Tp, rule=capacidad_volumen) # Sin 's'
        
        # (10 al 19)
        def capacidad_mano_obra(model, p, t):
            p_recep = self.map_p[self.name_recep]
            if p == p_recep: 
                return Constraint.Skip
            flujo_diario = sum(model.z[n, p, t, 0] for n in model.N if (n, p) in model.Pn)
            return flujo_diario <= model.CAP_labor_p[p]
        model.const_cap_man = Constraint(model.P, model.Tp, rule=capacidad_mano_obra)
        
        def capacidad_recepcion(model, t):
            p_recep = self.map_p[self.name_recep]
            carga_total = sum(model.q[a, t] + model.v[a, t] for a in model.A)
            return carga_total <= model.CAP_p[p_recep]
        model.const_cap_recep = Constraint(model.Tp, rule=capacidad_recepcion)
        
        def capacidad_recepcion_man(model, t):
            p_recep = self.map_p[self.name_recep]
            descarga_total = sum(model.q[a, t] for a in model.A)
            return descarga_total <= model.CAP_labor_p[p_recep]
        model.const_cap_recep_man = Constraint(model.Tp, rule=capacidad_recepcion_man)
        
        def min_lot_size(model, n, p, t): 
            return Q_min * model.delta[n, p, t] <= model.z[n, p, t, 0]
        model.const_min_lot = Constraint(model.Pn, model.Tp, rule=min_lot_size)
        
        def max_lot_size(model, n, p, t): 
            return model.z[n, p, t, 0] <= M * model.delta[n, p, t]
        model.const_max_lot = Constraint(model.Pn, model.Tp, rule=max_lot_size)
        
        def trucks(model, a, t): 
            return model.q[a,t] == L * model.l[a,t]
        model.const_trucks = Constraint(model.A, model.Tp, rule=trucks)

        def envejecimiento(model, n, p, t, u):
            tau_lim = int(value(model.Tau_np[n, p]))
            if u >= tau_lim - 1: 
                return Constraint.Skip
            if t == 1: 
                return model.z[n, p, t, u + 1] == model.gamma[n, p, u]          
            else: 
                return model.z[n, p, t, u + 1] == model.z[n, p, t-1, u]
        model.const_envej = Constraint(model.Pn, model.Tp, model.U, rule=envejecimiento)

        def flow_continuity(model, n, p, t):
            p_next = get_next_process(n, p)
            if p_next is None:                          
                return Constraint.Skip
            if (n, p_next) not in model.Pn:             
                return Constraint.Skip
            return model.z[n, p_next, t, 0] == model.x[n, p, p_next, t] 
        model.const_flow_continuity = Constraint(model.Pn, model.Tp, rule=flow_continuity)
        
        def entrada_mix(model, n, t):
            p_mix = self.map_p[self.name_mix]
            if (n, p_mix) not in model.Pn:              
                return Constraint.Skip
            return model.z[n, p_mix, t, 0] == model.y[n, t]
        model.const_ent_mix = Constraint(model.N, model.Tp, rule=entrada_mix)
        
        def receta(model, n, a, t): 
            return model.y[n, t] * model.R_ant[n, a, t] == model.w[n, a, t]
        model.const_receta = Constraint(model.An, model.Tp, rule=receta)

        return model.create_instance()
    
    def ExportarResultados(self, instance, nombre_archivo="Resultados_Determinista.xlsx"):
        def get_n(n_id): return self.Mezclas.iloc[n_id-1]['n']
        def get_a(a_id): return self.Alcoholes.iloc[a_id-1]['a']
        def get_p(p_id): return self.Procesos.iloc[p_id-1]['p']
        
        df_y = pd.DataFrame([
            (get_n(n), t, round(value(instance.y[n, t]), 2))
            for n in instance.N for t in instance.Tp 
            if value(instance.y[n, t]) > 0.001
        ], columns=["Mezcla", "Dia", "Litros_Producidos - y_nt"])

        times = instance.T if hasattr(instance, 'T') else instance.Tp
        df_v = pd.DataFrame([
            (get_a(a), t, round(value(instance.v[a, t]), 2)) 
            for a in instance.A for t in times
            if t <= self.N_Tiempo 
        ], columns=["Alcohol", "Dia", "Stock_Tanque - v_at"])

        df_k = pd.DataFrame([
            (get_n(n), get_p(p), t, round(value(instance.k[n, p, t]), 2)) 
            for (n,p) in instance.Pn for t in instance.Tp 
            if value(instance.k[n, p, t]) > 0.001
        ], columns=["Mezcla", "Proceso", "Dia", "Stock_Listo_k - k_npt"])
        
        df_i = pd.DataFrame([
            (get_n(n), t, round(value(instance.i_minus[n, t]), 2)) 
            for n in instance.N for t in instance.Tp
            if value(instance.i_minus[n, t]) > 0.001
        ], columns=["Mezcla", "Dia", "Backlog - i_minus_nt"])

        df_z = pd.DataFrame([
            (get_n(n), get_p(p), t, u, round(value(instance.z[n, p, t, u]), 2)) 
            for (n,p) in instance.Pn for t in instance.Tp for u in instance.U
            if value(instance.z[n, p, t, u]) > 0.001
        ], columns=["Mezcla", "Proceso", "Dia", "Edad", "Stock_Proceso_z - z_nupt"])

        df_q = pd.DataFrame([
            (get_a(a), t, round(value(instance.q[a, t]), 2)) 
            for a in instance.A for t in instance.Tp
            if value(instance.q[a, t]) > 0.001
        ], columns=["Alcohol", "Dia", "Recibido - q_at"])

        try:
            with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
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
        try:
            solver = create_gurobi_solver({"TimeLimit": 240})
            results = solver.solve(instance, tee=True)
        except GurobiConfigurationError as exc:
            print(f"\nError de configuración de Gurobi:\n{exc}\n")
            return instance
        
        if (results.solver.status == pyo.SolverStatus.ok) and \
           (results.solver.termination_condition == pyo.TerminationCondition.optimal):
            print('\n' + '='*50)
            print('SOLUCIÓN ÓPTIMA OPERATIVA ENCONTRADA')
            print('='*50)
            print(f'  - Costo I (Almacenamiento Alcohol):   $ {value(instance.coste_I):.2f}')
            print(f'  - Costo II (Inventario en Maduración):$ {value(instance.coste_II):.2f}')
            print(f'  - Costo III (Stock Terminado):        $ {value(instance.coste_III):.2f}')
            print(f'  - Costo IV (Penalización Backlog):    $ {value(instance.coste_IV):.2f}')
            print(f'COSTO TOTAL DIARIO (DETERMINISTA):  $ {value(instance.FO):.2f}')
            print('='*50 + '\n')

            self.ExportarResultados(instance)
        else:
            print("Error: No se encontró solución óptima.")
        return instance

if __name__ == "__main__":
    modelo = PiscoModel_Operativo()
    modelo.ReadExcelFile("Datos_Originales.xlsx")
    instancia = modelo.Solver()

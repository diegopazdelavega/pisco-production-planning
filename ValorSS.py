import pyomo.environ as pyo
from pyomo.environ import value, SolverFactory
from Pisco_Estc_Forecasting import PiscoModel
from Pisco_Det_Forecasting import PiscoModel_Operativo

def calcular_VSS(excel_path, pronostico_path):
    
# 1. calcular RP
    mod_stoc = PiscoModel()
    mod_stoc.ReadExcelFile(excel_path)
    mod_stoc.Cargar_Escenarios_Forecasting(
        ruta_csv=pronostico_path,
        dia_actual=0,
        num_escenarios=100, 
        alpha=0.05)
    
    inst_rp = mod_stoc.Solver()
    RP_value = value(inst_rp.FO)
    
# 2. calcular EV
    mod_det = PiscoModel_Operativo()
    mod_det.ReadExcelFile(excel_path)
    mod_det.Cargar_Pronostico_Determinista(
        ruta_csv=pronostico_path,
        dia_actual=0
    )
    
    inst_ev = mod_det.Solver()
    
    
# 3. calcular EEV
    mod_eev = PiscoModel()
    mod_eev.ReadExcelFile(excel_path)
    mod_eev.Cargar_Escenarios_Forecasting(
        ruta_csv=pronostico_path,
        dia_actual=0,
        num_escenarios=100, 
        alpha=0.05)
    
    # se crea la instancia pero sin solver
    inst_eev = mod_eev.Problema()
        
    # funcion para limpiar la basura de punto flotante de los solvers (ej: 1e-11 = 0.0)
    def limpiar_ruido(val):
        return 0.0 if val < 1e-4 else val

    # fijar variables independientes
    
    # 1. Recepción de Alcohol (se fija los lotes 'l' de camiones enteros)
    # 'q' (litros) se calculará solo a partir de 'l'
    for a in inst_eev.A:
        for t in inst_eev.Tp:
            inst_eev.l[a,t].fix(value(inst_ev.l[a,t]))
            
    # 2. Producción de Mezclas (se fija los litros preparados 'y')
    # 'w' (ingredientes) se calculará solo a partir de la receta de 'y'
    for n in inst_eev.N:
        for t in inst_eev.Tp:
            inst_eev.y[n,t].fix(limpiar_ruido(value(inst_ev.y[n,t])))

    # 3. Inventario Inicial Heredado ('b')
    for (n, p) in inst_eev.Pn:
        inst_eev.b[n,p].fix(limpiar_ruido(value(inst_ev.b[n,p])))

    # 4. Flujos entre procesos ('x')
    # Al fijar 'y', 'b' y 'x', las variables de maduración 'z' y el 
    # indicador de lote mínimo 'delta' se acomodarán solas.
    for n in inst_eev.N:
        for p in inst_eev.P:
            for d in inst_eev.P:
                for t in inst_eev.Tp:
                    inst_eev.x[n,p,d,t].fix(limpiar_ruido(value(inst_ev.x[n,p,d,t])))

    # Las variables de Segunda Etapa ('k', 'i_minus' y 'slack_cap') 
    # quedan libres para reaccionar a los escenarios de demanda

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 1800
    solver.options['MIPGap'] = 0.001
    results_eev = solver.solve(inst_eev, tee=False) 
    
    if (results_eev.solver.status == pyo.SolverStatus.ok) and \
       (results_eev.solver.termination_condition == pyo.TerminationCondition.optimal):
        
        EEV_value = value(inst_eev.FO)
        
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print(" RESULTADOS")
        print("="*60)
        print(f"Costo RP  (Modelo Estocástico Libre):   $ {RP_value:,.2f}")
        print(f"Costo EEV (Política Determinista real): $ {EEV_value:,.2f}")
        
        print("\n--- Desglose de Penalizaciones en EEV ---")
        print(f"  -> Costo IV (Multas por Quiebre/Atraso):   $ {value(inst_eev.coste_IV):,.2f}")
        
        if hasattr(inst_eev, 'coste_V'):
            print(f"  -> Costo V  (Multas por Sobrecapacidad):   $ {value(inst_eev.coste_V):,.2f}")
        
        VSS = EEV_value - RP_value
        ahorro_porcentual = (VSS / EEV_value) * 100 if EEV_value > 0 else 0
        
        print("\n" + "-" * 60)
        print(f"VSS:        $ {VSS:,.2f}")
        print(f"Ahorro relativo por usar RP en planta:     {ahorro_porcentual:.2f}%")
        print("="*60 + "\n")
        
    else:
        print("error.")


if __name__ == "__main__":
    ruta_excel = r"Datos_Originales.xlsx"
    pronostico = r"Evolucion_Pronosticos.xlsx"
    calcular_VSS(ruta_excel, pronostico)
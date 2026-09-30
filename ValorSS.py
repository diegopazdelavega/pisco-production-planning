import os
import pyomo.environ as pyo
from pyomo.environ import value, SolverFactory
import pandas as pd
import numpy as np
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
    solver.options['TimeLimit'] = 1800*2
    solver.options['MIPGap'] = 0.001
    results_eev = solver.solve(inst_eev, tee=False) 
    
    if (results_eev.solver.status == pyo.SolverStatus.ok) and \
       (results_eev.solver.termination_condition == pyo.TerminationCondition.optimal):
        
        EEV_value = value(inst_eev.FO)
        
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print(" RESULTADOS")
        print("="*60)
        print(f"Costo RP  (Modelo Estocástico Libre):   $ {RP_value:.2f}")
        print(f"Costo EEV (Política Determinista real): $ {EEV_value:.2f}")
        
        print("\n--- Desglose de Penalizaciones en EEV ---")
        print(f"  -> Costo IV (Multas por Quiebre/Atraso):   $ {value(inst_eev.coste_IV):.2f}")
        
        if hasattr(inst_eev, 'coste_V'):
            print(f"  -> Costo V  (Multas por Sobrecapacidad):   $ {value(inst_eev.coste_V):.2f}")
        
        VSS = EEV_value - RP_value
        ahorro_porcentual = (VSS / EEV_value) * 100 if EEV_value > 0 else 0
        
        print("\n" + "-" * 60)
        print(f"VSS:        $ {VSS:,.2f}")
        print(f"Ahorro relativo por usar RP en planta:     {ahorro_porcentual:.2f}%")
        print("="*60 + "\n")
        
    else:
        print("error.")


def calcular_VSS_realizado_oos(
    historial_det="Historial_Costos_RH_Determinista_OOS.xlsx",
    historial_est="Historial_Costos_RH_Estocastico_OOS.xlsx",
    nombre_salida="VSS_Realizado_OOS.xlsx",
):
    """
    CAMBIO OOS-VSS:
    Calcula el VSS Realizado usando los historiales out-of-sample de ambos RH.
    """
    df_det = pd.read_excel(historial_det, engine="openpyxl")
    df_est = pd.read_excel(historial_est, engine="openpyxl")

    columna_total = "Costo_Real_Total_RP"
    if columna_total not in df_det.columns or columna_total not in df_est.columns:
        raise ValueError(f"Los historiales deben contener la columna '{columna_total}'.")

    costo_det = float(df_det[columna_total].sum())
    costo_est = float(df_est[columna_total].sum())
    vss_realizado = costo_det - costo_est

    resumen = pd.DataFrame(
        [
            {
                "Costo_Real_Determinista": costo_det,
                "Costo_Real_Estocastico": costo_est,
                "VSS_Realizado_OOS": vss_realizado,
            }
        ]
    )
    resumen.to_excel(nombre_salida, index=False, engine="openpyxl")

    print("\n" + "=" * 60)
    print(" VSS REALIZADO OUT-OF-SAMPLE")
    print("=" * 60)
    print(f"Costo real RH Determinista: $ {costo_det:,.2f}")
    print(f"Costo real RH Estocástico:  $ {costo_est:,.2f}")
    print(f"VSS Realizado OOS:          $ {vss_realizado:,.2f}")
    costo_total_det = pd.read_excel("Historial_Costos_RH_Determinista_OOS.xlsx", engine="openpyxl")["Costo_Real_Total_RP"].sum()
    costo_total_est = pd.read_excel("Historial_Costos_RH_Estocastico_OOS.xlsx", engine="openpyxl")["Costo_Real_Total_RP"].sum()

    vss_pct = ((costo_total_det - costo_total_est) / costo_total_det) * 100
    print(f"VSS OOS porcentual: {vss_pct:.2f}%")
    print(f"Resumen exportado a: {nombre_salida}")
    print("=" * 60 + "\n")

    return vss_realizado

#---- helper: VSS in-sample con semilla configurable ----
def calcular_VSS_con_semilla(excel_path, pronostico_path, semilla=51):

    # 1) Estocástico (RP)
    mod_stoc = PiscoModel()
    mod_stoc.ReadExcelFile(excel_path)
    mod_stoc.Cargar_Escenarios_Forecasting(
        ruta_csv=pronostico_path, dia_actual=0, num_escenarios=100, alpha=0.05, semilla=semilla
    )
    inst_rp = mod_stoc.Solver()
    RP_value = value(inst_rp.FO)

    # 2) Determinista base (EV)
    mod_det = PiscoModel_Operativo()
    mod_det.ReadExcelFile(excel_path)
    mod_det.Cargar_Pronostico_Determinista(ruta_csv=pronostico_path, dia_actual=0)
    inst_ev = mod_det.Solver()
    
    # NUEVO: Capturar el valor esperado (EV) del plan determinista
    EV_value = value(inst_ev.FO)

    # 3) EEV (fija la política determinista y resuelve bajo escenarios)
    mod_eev = PiscoModel()
    mod_eev.ReadExcelFile(excel_path)
    mod_eev.Cargar_Escenarios_Forecasting(
        ruta_csv=pronostico_path, dia_actual=0, num_escenarios=100, alpha=0.05, semilla=semilla
    )
    inst_eev = mod_eev.Problema()

    def limpiar_ruido(val):
        return 0.0 if val < 1e-4 else val

    # Fijar variables de 1ra etapa
    for a in inst_eev.A:
        for t in inst_eev.Tp:
            inst_eev.l[a, t].fix(value(inst_ev.l[a, t]))
    for n in inst_eev.N:
        for t in inst_eev.Tp:
            inst_eev.y[n, t].fix(limpiar_ruido(value(inst_ev.y[n, t])))
    for (n, p) in inst_eev.Pn:
        inst_eev.b[n, p].fix(limpiar_ruido(value(inst_ev.b[n, p])))
    for n in inst_eev.N:
        for p in inst_eev.P:
            for d in inst_eev.P:
                for t in inst_eev.Tp:
                    inst_eev.x[n, p, d, t].fix(limpiar_ruido(value(inst_ev.x[n, p, d, t])))

    solver = SolverFactory("gurobi")
    solver.options["TimeLimit"] = 1800 * 2
    solver.options["MIPGap"] = 0.001
    results_eev = solver.solve(inst_eev, tee=False)

    if not (results_eev.solver.status == pyo.SolverStatus.ok and 
            results_eev.solver.termination_condition == pyo.TerminationCondition.optimal):
        raise RuntimeError(f"No convergió EEV para semilla {semilla}")

    EEV_value = value(inst_eev.FO)
    VSS = EEV_value - RP_value
    ahorro_pct = (VSS / EEV_value) if EEV_value > 0 else 0.0 # Se quita el *100 para formato Excel

    return {
        "EV": EV_value,
        "RP": RP_value,
        "EEV": EEV_value,
        "VSS_in_sample": VSS,
        "VSS_in_sample_pct": ahorro_pct
    }

def ejecutar_sensibilidad_vss():
    semillas = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
    filas = []

    # 1. CARGAR LÍNEA BASE DETERMINISTA (ÚNICA)
    archivo_det_base = "Historial_Costos_RH_Determinista_OOS.xlsx"
    if not os.path.exists(archivo_det_base):
        print(f"[ERROR] No se encuentra la línea base: {archivo_det_base}")
        return pd.DataFrame()
        
    df_det = pd.read_excel(archivo_det_base, engine="openpyxl")
    costo_det_dinamico = float(df_det["Costo_Real_Total_RP"].sum())
    
    df_det_iter0 = df_det[df_det["Iteracion"] == 0]
    costo_det_estatico = float(df_det_iter0["Costo_Real_Total_RP"].sum()) if not df_det_iter0.empty else 0.0

    # 2. BUCLE POR CADA SEMILLA ESTOCÁSTICA
    for i, semilla in enumerate(semillas):
        print(f"\nProcesando Simulación {i+1} (Semilla {semilla})...")
        
        # --- A. CÁLCULO IN-SAMPLE (EV, RP, EEV) ---
        try:
            res_in = calcular_VSS_con_semilla(
                excel_path="Datos_Originales.xlsx", pronostico_path="Evolucion_Pronosticos.xlsx", semilla=semilla
            )
        except Exception as e:
            print(f"[WARN] Error al calcular VSS IS para semilla {semilla}: {e}")
            res_in = {}

        # --- B. EXTRACCIÓN OUT-OF-SAMPLE (Realidad) ---
        nombre_est = f"Historial_Costos_RH_Estocastico_OOS_semilla_{semilla}.xlsx"
        if not os.path.exists(nombre_est):
            print(f"[WARN] Falta archivo OOS para semilla {semilla}. Se omite.")
            continue

        df_est = pd.read_excel(nombre_est, engine="openpyxl")
        
        # OOS Dinámico (Total del año)
        costo_est_dinamico = float(df_est["Costo_Real_Total_RP"].sum())
        vss_oos_dinamico = costo_det_dinamico - costo_est_dinamico
        vss_oos_din_pct = (vss_oos_dinamico / costo_det_dinamico) if costo_det_dinamico > 0 else 0.0
        
        # OOS Estático (Solo mes 1 / Iteración 0)
        df_est_iter0 = df_est[df_est["Iteracion"] == 0]
        costo_est_estatico = float(df_est_iter0["Costo_Real_Total_RP"].sum()) if not df_est_iter0.empty else 0.0
        vss_oos_estatico = costo_det_estatico - costo_est_estatico
        vss_oos_est_pct = (vss_oos_estatico / costo_det_estatico) if costo_det_estatico > 0 else 0.0

        # --- CONSTRUCCIÓN DE LA FILA (FORMATO TESIS) ---
        filas.append({
            "Simulación": i + 1,
            "Semilla": semilla,
            "Escenarios": 100,
            # BLOQUE IN-SAMPLE (Copia de tu imagen)
            "EV (Det base)": res_in.get("EV", 0),
            "RP (GAP 0.001)": res_in.get("RP", 0),
            "EEV (GAP 0.001)": res_in.get("EEV", 0),
            "VSS IS": res_in.get("VSS_in_sample", 0),
            "VSS IS %": res_in.get("VSS_in_sample_pct", 0),
            # BLOQUE OUT-OF-SAMPLE ESTÁTICO (Iteración 0)
            "Det OOS Estático": costo_det_estatico,
            "Estoc OOS Estático": costo_est_estatico,
            "VSS OOS Estático": vss_oos_estatico,
            "VSS OOS Estático %": vss_oos_est_pct,
            # BLOQUE OUT-OF-SAMPLE DINÁMICO (Total Año)
            "Det OOS Dinámico": costo_det_dinamico,
            "Estoc OOS Dinámico": costo_est_dinamico,
            "VSS OOS Dinámico": vss_oos_dinamico,
            "VSS OOS Dinámico %": vss_oos_din_pct
        })

    # 3. CREAR DATAFRAME Y CALCULAR FILA DE PROMEDIOS
    df_resumen = pd.DataFrame(filas)
    
    fila_promedio = {
        "Simulación": "",
        "Semilla": "PROMEDIO",
        "Escenarios": 100,
        "EV (Det base)": df_resumen["EV (Det base)"].mean(),
        "RP (GAP 0.001)": df_resumen["RP (GAP 0.001)"].mean(),
        "EEV (GAP 0.001)": df_resumen["EEV (GAP 0.001)"].mean(),
        "VSS IS": df_resumen["VSS IS"].mean(),
        "VSS IS %": df_resumen["VSS IS %"].mean(),
        "Det OOS Estático": df_resumen["Det OOS Estático"].mean(),
        "Estoc OOS Estático": df_resumen["Estoc OOS Estático"].mean(),
        "VSS OOS Estático": df_resumen["VSS OOS Estático"].mean(),
        "VSS OOS Estático %": df_resumen["VSS OOS Estático %"].mean(),
        "Det OOS Dinámico": df_resumen["Det OOS Dinámico"].mean(),
        "Estoc OOS Dinámico": df_resumen["Estoc OOS Dinámico"].mean(),
        "VSS OOS Dinámico": df_resumen["VSS OOS Dinámico"].mean(),
        "VSS OOS Dinámico %": df_resumen["VSS OOS Dinámico %"].mean()
    }
    
    # Agregar la fila de promedios al final usando concat (método moderno de Pandas)
    df_resumen = pd.concat([df_resumen, pd.DataFrame([fila_promedio])], ignore_index=True)

    # 4. EXPORTAR A EXCEL
    nombre_salida = "Reporte_Consolidado_VSS_Sensibilidad.xlsx"
    df_resumen.to_excel(nombre_salida, index=False, engine="openpyxl")
    
    print("\n" + "="*70)
    print(f" REPORTE EXPORTADO EXITOSAMENTE A: {nombre_salida}")
    print("="*70)
    print("El archivo contiene todas las semillas desglosadas por IS y OOS,")
    print("incluyendo una fila final con los promedios totales calculados.\n")

    return df_resumen


if __name__ == "__main__":

    ruta_excel = r"Datos_Originales.xlsx"
    pronostico = r"Evolucion_Pronosticos.xlsx"

    print("\n=== VSS IN-SAMPLE (CÁLCULO TEÓRICO) ===")
    vss_in = calcular_VSS(ruta_excel, pronostico)

    print("\n=== SENSIBILIDAD POR SEMILLAS Y VSS REALIZADO (OOS) ===")
    ejecutar_sensibilidad_vss()




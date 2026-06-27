import pandas as pd
import numpy as np
from collections import defaultdict

def cargar_demanda_determinista(ruta_csv, dia_actual):
    """
    Filtra el Excel de pronósticos para el 'dia_actual' y extrae F_t.
    """
    df = pd.read_excel(ruta_csv)
    
    # Pronostico del dia actual: filtram por 'Fecha de Hoy' == dia_actual
    df_hoy = df[df['Fecha de Hoy'] == dia_actual]
    
    if df_hoy.empty:
        print(f"No hay pronósticos para el Día {dia_actual}.")
    
    # defaultdict devuelve 0, en caso de que no haya pronóstico para una mezcla y día específico
    demanda_dict = defaultdict(float)
    
    for index, row in df_hoy.iterrows():
        mezcla = row['n']
        
        # El día real en que el cliente espera el producto (Día de despacho)
        # Es la suma del día de hoy + los días que faltan (t)
        dia_entrega = int(row['Fecha de Hoy'] + row['t'])
        
        pronostico_F_t = max(0, row['F_t']) 
        
        # Asignamos al diccionario (n, dia)
        demanda_dict[(mezcla, dia_entrega)] = pronostico_F_t
        
    return dict(demanda_dict)


def cargar_demanda_estocastica(ruta_csv, dia_actual, num_escenarios=1, alpha=0.05):
    """
    Filtra el Excel para el 'dia_actual', lee la base del pronóstico (v_t) 
    y genera 'W' escenarios usando la dispersión de Clark.
    """
    df = pd.read_excel(ruta_csv)
    df_hoy = df[df['Fecha de Hoy'] == dia_actual]
    
    demanda_dict = defaultdict(float)
    
    # Fijamos una semilla dinámica basada en el día para reproducibilidad
    np.random.seed(42)
    
    for index, row in df_hoy.iterrows():
        mezcla = row['n']
        dia_entrega = int(row['Fecha de Hoy'] + row['t'])
        
        # Para generar escenarios, tomamos la base (v_t) y el multiplicador temporal (t)
        # v_base = row['v_t']
        tiempo_restante = row['t']
                
        # CASO 1: La demanda es para HOY (t = 0). Es 100% conocida.
        if tiempo_restante == 0:
            for s in range(1, num_escenarios + 1):
                # Todos los escenarios reciben exactamente el mismo valor real (sin ruido)
                demanda_dict[(mezcla, dia_entrega, s)] = row['v_t']
                
        # CASO 2: La demanda es FUTURA (t > 0). Hay incertidumbre y escenarios.
        else:
            media = row['F_t']                                     # El centro de la distribución (Media) es el PRONÓSTICO DE HOY (F_t)
            desviacion_est = abs(media * alpha * tiempo_restante)  # Desviación estándar proporcional al pronóstico y al tiempo restante, leadtime

            escenarios = np.random.normal(loc=media, scale=desviacion_est, size=num_escenarios) # genera los escenarios de demanda en base a la distribución normal centrada en F_t

            for s in range(1, num_escenarios + 1):
                r = np.random.normal(0, 1)
                
                # Ecuación 27 de Clark para escenarios: F_tw = v_t * (1 + t * alpha * r)
                # F_tw = v_base * (1 + tiempo_restante * alpha * r)
                
                # Asignamos al diccionario tridimensional (n, dia, s)
                demanda_dict[(mezcla, dia_entrega, s)] = max(0, escenarios[s-1])  # Aseguramos que la demanda no sea negativa
            
    return dict(demanda_dict)

# =====================================================================
# EJEMPLO DE CÓMO INYECTARLO EN TU BUCLE DE ROLLING HORIZON
# ====================================================================

# =====================================================================
# EJEMPLO DE CÓMO INYECTARLO EN TU BUCLE DE ROLLING HORIZON
# =====================================================================
if __name__ == "__main__":
    ruta_archivo = "Evolucion_Pronosticos.xlsx"
    
    # Supongamos que arranca el bucle en la Iteración 1 (Día 30)
    dia_de_iteracion = 30 
    
    # 1. Para Determinista:
    dict_det = cargar_demanda_determinista(ruta_archivo, dia_actual=dia_de_iteracion)
    
    # 2. Para Estocástico:
    dict_est = cargar_demanda_estocastica(ruta_archivo, dia_actual=dia_de_iteracion, num_escenarios=1, alpha=0.05)
    
    # --- MÉTODO 2: EXPORTAR A EXCEL PARA COMPROBACIÓN VISUAL ---
    print("\nGenerando Excel de comprobación...")
    df_validacion = pd.DataFrame([
        {'Mezcla': key[0], 'Dia_Entrega': key[1], 'Escenario': key[2], 'Demanda_Generada': val}
        for key, val in dict_est.items()
    ])
    df_validacion.to_excel("Validacion_Escenarios.xlsx", index=False)
    print("¡Listo! Revisa el archivo 'Validacion_Escenarios.xlsx'.")
    print("Módulos de integración listos para Pyomo.")

    
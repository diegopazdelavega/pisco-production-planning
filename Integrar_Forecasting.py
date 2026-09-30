import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# CAMBIO OOS-VSS: columnas aceptadas para la demanda real out-of-sample.
COLUMNAS_DEMANDA_REAL = ("V_0", "v0", "V0", "v_0", "v_t")


def _normalizar_valor_no_negativo(valor):
    return max(0.0, float(valor))


def _columna_demanda_real(df):
    for columna in COLUMNAS_DEMANDA_REAL:
        if columna in df.columns:
            return columna
    columnas = ", ".join(COLUMNAS_DEMANDA_REAL)
    raise ValueError(f"No se encontró columna de demanda real. Columnas esperadas: {columnas}")


def factor_incertidumbre_no_lineal(tiempo_restante):
    """
    Es 0 cuando t=0 y crece lentamente para horizontes lejanos.
    """
    tiempo_restante = max(0.0, float(tiempo_restante))
    return np.sqrt(np.log1p(tiempo_restante)) #raiz(log(1+t))


def calcular_desviacion_clark(media, tiempo_restante, alpha=0.05, limite_sigma=4):
    """
    Desviacion estandar Clark refinada:
    sigma = |media| * alpha * sqrt(log(t + 1)).

    El tope media / limite_sigma evita que el limite inferior media - 4*sigma
    sea negativo cuando se usa limite_sigma=4.
    """
    media = max(0.0, float(media)) #F_t
    sigma = media * alpha * factor_incertidumbre_no_lineal(tiempo_restante) #F_t*alpha*raiz(log(1+t)) desv est

    if limite_sigma and limite_sigma > 0 and media > 0:
        sigma = min(sigma, media / limite_sigma) # media/limite_sigma evita que el limite inferior media - 4*sigma sea negativo cuando se usa limite_sigma=4

    return sigma


def generar_muestras_clark(media, tiempo_restante, num_muestras, alpha=0.05, limite_sigma=4, rng=None):
    """
    Genera muestras normales centradas en la media entregada, con disipacion
    no lineal de incertidumbre y truncamiento practico a +/- limite_sigma.
    """
    if rng is None:
        rng = np.random.default_rng()

    media = max(0.0, float(media)) #F_t
    sigma = calcular_desviacion_clark(media, tiempo_restante, alpha, limite_sigma)

    if sigma == 0:
        return np.full(num_muestras, media)

    muestras = rng.normal(loc=media, scale=sigma, size=num_muestras) #generar escenarios con media y desv estandar

    if limite_sigma and limite_sigma > 0:
        amplitud_maxima = limite_sigma * sigma
        muestras = np.clip(muestras, media - amplitud_maxima, media + amplitud_maxima) # intervalo de confianza de 4 sigmas (limite_sigma=4) para evitar valores negativos
    return np.maximum(0.0, muestras)


def cargar_demanda_determinista(ruta_csv, dia_actual):
    """
    Filtra el Excel de pronósticos para el 'dia_actual' y extrae F_t.
    """
    df = pd.read_excel(ruta_csv, engine='openpyxl')
    
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
        
        # Si t=0, la demanda es conocida y es v_t (que es igual a v0).
        # Si t>0, la demanda es el pronóstico F_t.
        if row['t'] == 0:
            demanda_real = max(0, row['v_t'])
            demanda_dict[(mezcla, dia_entrega)] = demanda_real
        else:
            pronostico_F_t = max(0, row['F_t']) 
            demanda_dict[(mezcla, dia_entrega)] = pronostico_F_t
        # Asignamos al diccionario (n, dia)
    return dict(demanda_dict)


def cargar_demanda_estocastica(ruta_csv, dia_actual, num_escenarios=1, alpha=0.05, rng=None):
    """
    Filtra el Excel para el 'dia_actual', lee la base del pronóstico (v_t) 
    y genera s escenarios.
    """
    df = pd.read_excel(ruta_csv, engine='openpyxl')
    df_hoy = df[df['Fecha de Hoy'] == dia_actual]
    
    demanda_dict = defaultdict(float)
    if rng is None:
        rng = np.random.default_rng()
    
    for index, row in df_hoy.iterrows():
        mezcla = row['n']
        dia_entrega = int(row['Fecha de Hoy'] + row['t'])
        
        # Para generar escenarios, toma la base (v_t) y el multiplicador temporal (t)
        # v_base = row['v_t']
        tiempo_restante = row['t']
                
        # 1. La demanda es para hoy (t = 0). Es 100% conocida.
        if tiempo_restante == 0:
            for s in range(1, num_escenarios + 1):
                # Todos los escenarios reciben exactamente el mismo valor real (sin ruido)
                demanda_dict[(mezcla, dia_entrega, s)] = row['v_t']
                
        # 2. La demanda tiene leadtime (t > 0). incertidumbre y escenarios.
        else:
            media = row['F_t']                                     # El centro de la distribución (Media) es el PRONÓSTICO DE HOY (F_t)
            escenarios = generar_muestras_clark(
                media=media,
                tiempo_restante=tiempo_restante,
                num_muestras=num_escenarios,
                alpha=alpha,
                rng=rng
            )

            for s in range(1, num_escenarios + 1):
                # diccionario  (n, dia, s)
                demanda_dict[(mezcla, dia_entrega, s)] = escenarios[s-1]  # Aseguramos que la demanda no sea negativa
            
    return dict(demanda_dict)


def cargar_demanda_real(ruta_csv, dia_actual=None, horizonte=None):
    """
    CAMBIO OOS-VSS:
    Carga la realidad out-of-sample V_0/v0 como unica fuente de verdad.

    Devuelve un diccionario determinista:
        (mezcla, dia_entrega_absoluto) -> demanda_real_v0

    Si se entrega dia_actual y horizonte, filtra solo los despachos del tramo
    (dia_actual, dia_actual + horizonte].
    """
    df = pd.read_excel(ruta_csv, engine='openpyxl')
    columna_real = _columna_demanda_real(df)
    demanda_dict = defaultdict(float)

    for _, row in df.iterrows():
        mezcla = row['n']
        dia_entrega = int(row['Fecha de Hoy'] + row['t'])

        if dia_actual is not None and horizonte is not None:
            if dia_entrega <= dia_actual or dia_entrega > dia_actual + horizonte:
                continue

        demanda_dict[(mezcla, dia_entrega)] = _normalizar_valor_no_negativo(row[columna_real])

    return dict(demanda_dict)

def cargar_escenario_101(ruta_csv, dia_actual, alpha=0.05, semilla_realidad=1001):
    """
    Genera el Escenario Real Sintético (OOS). 
    Usa una semilla exclusiva para garantizar que la realidad sea 
    impredecible e independiente de los escenarios de planificación.
    Devuelve un diccionario 2D: (mezcla, dia_entrega) -> valor
    """
    # 1. Semilla aislada que avanza de forma única con el tiempo
    rng_realidad = np.random.default_rng(semilla_realidad + int(dia_actual))
    
    # 2. Reutilizamos tu función estocástica base, pidiendo 1 solo escenario
    dict_estocastico = cargar_demanda_estocastica(
        ruta_csv=ruta_csv, 
        dia_actual=dia_actual, 
        num_escenarios=1, 
        alpha=alpha, 
        rng=rng_realidad
    )
    
    # 3. Aplanamos la clave de 3D (mezcla, dia, s) a 2D (mezcla, dia)
    demanda_101 = {}
    for (mezcla, dia_entrega, s), valor in dict_estocastico.items():
        demanda_101[(mezcla, dia_entrega)] = valor
        
    return demanda_101

if __name__ == "__main__":
    ruta_archivo = "Evolucion_Pronosticos.xlsx"
    
    # día 0 de la simulación
    dia_de_iteracion = 0 
    
    # 1. Para Determinista:
    dict_det = cargar_demanda_determinista(ruta_archivo, dia_actual=dia_de_iteracion)
    
    # dict det
    print("Diccionario Determinista")
    print("(Mezcla, Dia Entrega) : Pronostico ")
    for i, (key, value) in enumerate(dict_det.items()):
        if i >= 5:
            break
        print(f"{key} -> {value:.2f}")

    # 2. Para Estocástico:
    rng = np.random.default_rng(51)
    dict_est = cargar_demanda_estocastica(
        ruta_archivo,
        dia_actual=dia_de_iteracion,
        num_escenarios=5,
        alpha=0.05,
        rng=rng
    )
    
    # dict estc
    print("Diccionario Estocástico")
    print("(Mezcla, Dia Entrega, Escenario) : Pronostico")
    for i, (key, value) in enumerate(dict_est.items()):
        if i >= 5:
            break
        print(f"{key} -> {value:.2f}")

    # excel para diccionario estocástico completo
    if dict_est:
        df_validacion = pd.DataFrame([
            {'Mezcla': key[0], 'Dia_Entrega': key[1], 'Escenario': key[2], 'Demanda_Generada': val}
            for key, val in dict_est.items()
        ])
        df_validacion.to_excel("Validacion_Escenarios.xlsx", index=False)
    print("\nInspección completada.")

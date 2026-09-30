import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"Fecha de Hoy", "n", "t", "v_t", "F_t"}


def _normalizar_valor_no_negativo(valor):
    """Convierte valores a float y evita negativos."""
    return max(0.0, float(valor))


def _validar_columnas(df):
    """Valida que el Excel tenga las columnas requeridas."""
    columnas_faltantes = REQUIRED_COLUMNS - set(df.columns)
    if columnas_faltantes:
        faltantes = ", ".join(sorted(columnas_faltantes))
        raise ValueError(f"Faltan columnas requeridas en el archivo de pronóstico: {faltantes}")
    return df


def _preparar_df_pronostico(ruta_csv, dia_actual):
    """Lee el Excel y filtra solo los registros del día actual."""
    df = pd.read_excel(ruta_csv, engine="openpyxl")
    df = _validar_columnas(df)

    df_filtrado = df[df["Fecha de Hoy"] == dia_actual].copy()
    if df_filtrado.empty:
        print(f"No hay pronósticos para el Día {dia_actual}.")

    df_filtrado["t"] = pd.to_numeric(df_filtrado["t"], errors="coerce").fillna(0).astype(int)
    df_filtrado["v_t"] = pd.to_numeric(df_filtrado["v_t"], errors="coerce").fillna(0.0)
    df_filtrado["F_t"] = pd.to_numeric(df_filtrado["F_t"], errors="coerce").fillna(0.0)
    return df_filtrado


def factor_incertidumbre_no_lineal(tiempo_restante):
    """
    Escala la incertidumbre con una función sublineal.
    Es 0 cuando t=0 y crece lentamente para horizontes lejanos.
    """
    tiempo_restante = max(0.0, float(tiempo_restante))
    return np.sqrt(np.log1p(tiempo_restante))


def calcular_desviacion_clark(media, tiempo_restante, alpha=0.05, limite_sigma=4):
    """
    Desviación estándar Clark refinada:
    sigma = |media| * alpha * sqrt(log(t + 1)).

    El tope media / limite_sigma evita que el límite inferior media - 4*sigma
    sea negativo cuando se usa limite_sigma=4.
    """
    media = _normalizar_valor_no_negativo(media)
    sigma = media * alpha * factor_incertidumbre_no_lineal(tiempo_restante)

    if limite_sigma and limite_sigma > 0 and media > 0:
        sigma = min(sigma, media / limite_sigma)

    return sigma


def generar_muestras_clark(media, tiempo_restante, num_muestras, alpha=0.05, limite_sigma=4, rng=None):
    """
    Genera muestras normales centradas en la media entregada, con disipación
    no lineal de incertidumbre y truncamiento práctico a +/- limite_sigma.
    """
    if rng is None:
        rng = np.random.default_rng()

    media = _normalizar_valor_no_negativo(media)
    sigma = calcular_desviacion_clark(media, tiempo_restante, alpha, limite_sigma)

    if sigma == 0:
        return np.full(num_muestras, media, dtype=float)

    muestras = rng.normal(loc=media, scale=sigma, size=num_muestras)

    if limite_sigma and limite_sigma > 0:
        amplitud_maxima = limite_sigma * sigma
        muestras = np.clip(muestras, media - amplitud_maxima, media + amplitud_maxima)

    return np.maximum(0.0, muestras).astype(float)


def cargar_demanda_determinista(ruta_csv, dia_actual):
    """
    Filtra el Excel de pronósticos para el día actual y extrae la demanda F_t
    como un diccionario de la forma (mezcla, dia_entrega) -> demanda.
    """
    df_hoy = _preparar_df_pronostico(ruta_csv, dia_actual)
    demanda_dict = {}

    for record in df_hoy.to_dict("records"):
        mezcla = record["n"]
        t = int(record["t"])
        dia_hoy = int(record["Fecha de Hoy"])
        dia_entrega = dia_hoy + t

        if t == 0:
            demanda = _normalizar_valor_no_negativo(record["v_t"])
        else:
            demanda = _normalizar_valor_no_negativo(record["F_t"])

        demanda_dict[(mezcla, dia_entrega)] = demanda

    return demanda_dict


def cargar_demanda_estocastica(ruta_csv, dia_actual, num_escenarios=1, alpha=0.05, rng=None):
    """
    Filtra el Excel para el día actual e integra la demanda pronosticada con
    escenarios aleatorios. Devuelve un diccionario con clave:
    (mezcla, dia_entrega, escenario).
    """
    df_hoy = _preparar_df_pronostico(ruta_csv, dia_actual)
    if rng is None:
        rng = np.random.default_rng()

    demanda_dict = {}

    for record in df_hoy.to_dict("records"):
        mezcla = record["n"]
        t = int(record["t"])
        dia_hoy = int(record["Fecha de Hoy"])
        dia_entrega = dia_hoy + t

        if t == 0:
            valor_conocido = _normalizar_valor_no_negativo(record["v_t"])
            for escenario in range(1, num_escenarios + 1):
                demanda_dict[(mezcla, dia_entrega, escenario)] = valor_conocido
        else:
            media = _normalizar_valor_no_negativo(record["F_t"])
            escenarios = generar_muestras_clark(
                media=media,
                tiempo_restante=t,
                num_muestras=num_escenarios,
                alpha=alpha,
                rng=rng,
            )

            for escenario, valor in enumerate(escenarios, start=1):
                demanda_dict[(mezcla, dia_entrega, escenario)] = float(valor)

    return demanda_dict


def _validar_ejecucion_demo():
    """Ejemplo de verificación rápida del comportamiento del script."""
    ruta_archivo = "Evolucion_Pronosticos.xlsx"
    dia_de_iteracion = 0

    dict_det = cargar_demanda_determinista(ruta_archivo, dia_actual=dia_de_iteracion)
    print("Diccionario Determinista")
    print("(Mezcla, Dia Entrega) : Pronostico")
    for i, (key, value) in enumerate(dict_det.items()):
        if i >= 5:
            break
        print(f"{key} -> {value:.2f}")

    rng = np.random.default_rng(51)
    dict_est = cargar_demanda_estocastica(
        ruta_archivo,
        dia_actual=dia_de_iteracion,
        num_escenarios=3,
        alpha=0.05,
        rng=rng,
    )

    print("\nDiccionario Estocástico")
    print("(Mezcla, Dia Entrega, Escenario) : Pronostico")
    for i, (key, value) in enumerate(dict_est.items()):
        if i >= 5:
            break
        print(f"{key} -> {value:.2f}")

    if dict_est:
        df_validacion = pd.DataFrame(
            [
                {"Mezcla": key[0], "Dia_Entrega": key[1], "Escenario": key[2], "Demanda_Generada": val}
                for key, val in dict_est.items()
            ]
        )
        df_validacion.to_excel("Validacion_Escenarios.xlsx", index=False)

    print("\nInspección completada.")


if __name__ == "__main__":
    _validar_ejecucion_demo()

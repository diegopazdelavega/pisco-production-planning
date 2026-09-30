import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Integrar_Forecasting import generar_muestras_clark


def generar_escenarios_desde_base(
    ruta_excel="Evolucion_Pronosticos.xlsx",
    semilla_base=51,
    num_escenarios=6,
    alpha=0.05,
    mezcla='n1',
    periodo_T=16,
):
    """
    Usa EXACTAMENTE la misma lógica que Integrar_Forecasting:
    - lee la base del pronóstico ya generada,
    - toma F_t como media,
    - genera escenarios con el mismo RNG y la misma función de incertidumbre.
    """
    df_base = pd.read_excel(ruta_excel, engine='openpyxl')
    if 'Semilla' not in df_base.columns:
        df_base['Semilla'] = semilla_base

    df_plot = df_base[(df_base['n'] == mezcla) & (df_base['T'] == periodo_T)].copy()

    if df_plot.empty:
        raise ValueError(f"No se encontraron datos para mezcla={mezcla} y T={periodo_T}.")

    df_plot = df_plot.sort_values(['Semilla', 't'], ascending=[True, False]).reset_index(drop=True)

    rng = np.random.default_rng(semilla_base)
    filas = []

    for _, row in df_plot.iterrows():
        n = row['n']
        T = int(row['T'])
        t = int(row['t'])
        v0 = float(row['v0'])
        v_t = float(row['v_t'])
        f_t_base = float(row['F_t'])

        escenarios = generar_muestras_clark(
            media=f_t_base,
            tiempo_restante=t,
            num_muestras=num_escenarios,
            alpha=alpha,
            rng=rng,
        )

        for escenario_id in range(1, num_escenarios + 1):
            valor = float(escenarios[escenario_id - 1])
            filas.append({
                'Semilla': semilla_base,
                'Escenario': escenario_id,
                'n': n,
                'T': T,
                't': t,
                'v0': v0,
                'v_t': v_t,
                'F_t_base': f_t_base,
                'F_t_escenario': valor,
                'Demanda_Generada': valor,
                'r_t': valor - f_t_base,
            })

    return pd.DataFrame(filas)


def graficar_trayectorias_por_escenario(df, mezcla='n1', periodo_T=16):
    """Grafica cada trayectoria generada con la misma base del integrador del forecast."""
    df_plot = df[(df['n'] == mezcla) & (df['T'] == periodo_T)].copy()
    if df_plot.empty:
        print(f"No hay datos para mezcla={mezcla}, T={periodo_T}")
        return

    df_plot = df_plot.sort_values(['Semilla', 'Escenario', 't'], ascending=[True, True, False]).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    v0_real = df_plot['v0'].iloc[0]

    for escenario in sorted(df_plot['Escenario'].unique()):
        df_esc = df_plot[df_plot['Escenario'] == escenario].sort_values('t', ascending=False)
        plt.plot(
            df_esc['t'],
            df_esc['Demanda_Generada'],
            marker='o',
            markersize=4,
            linestyle='-',
            linewidth=1.5,
            label=f'Escenario {escenario}',
        )

    media_base = df_plot.groupby('t', as_index=False)['F_t_base'].mean()
    media_base = media_base.sort_values('t', ascending=False)
    plt.plot(
        media_base['t'],
        media_base['F_t_base'],
        linestyle='--',
        linewidth=2.5,
        color='black',
        alpha=0.9,
        label='Media F_t (base)',
    )

    plt.axhline(y=v0_real, color='green', linewidth=2.5, linestyle='-', label=f'v0 = {v0_real}')
    plt.gca().invert_xaxis()

    plt.title(
        f"Trayectorias de demanda estocástica usando la misma base de Integrar_Forecasting\n"
        f"Mezcla: {mezcla} | T de Entrega: {periodo_T}",
        fontsize=14,
        fontweight='bold',
        pad=15,
    )
    plt.xlabel('Periodos restantes antes de la demanda real (t)', fontsize=12)
    plt.ylabel('Demanda (Unidades)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_escenarios = generar_escenarios_desde_base(
        ruta_excel='Evolucion_Pronosticos.xlsx',
        semilla_base=51,
        num_escenarios=5,
        alpha=0.05,
        mezcla='n1',
        periodo_T=16,
    )

    print(df_escenarios.head(10).to_string(index=False))
    graficar_trayectorias_por_escenario(df_escenarios, mezcla='n1', periodo_T=16)

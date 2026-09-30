import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Integrar_Forecasting import calcular_desviacion_clark, factor_incertidumbre_no_lineal

df_demands = pd.read_excel('Datos_Originales.xlsx', sheet_name='Demands')
df_demands.rename(columns={'n': 'n', 't': 't', 'D': 'v0'}, inplace=True)

# Para correr más iteraciones, necesitamos pronósticos más lejanos.
# duplicar las demandas existentes y empujarlas 210 días hacia el futuro.
df_demands_future = df_demands.copy()
df_demands_future['t'] = df_demands_future['t'] + 210 # Empujamos el tiempo de entrega

# Unimos la demanda original con la futura
df_demands_extended = pd.concat([df_demands, df_demands_future], ignore_index=True)

alpha = 0.05

# --- 2. FUNCIÓN DE SIMULACIÓN ---
def simular_escenario(df_base, valor_alpha, semilla, rng=None):
    if rng is None:
        rng = np.random.default_rng(semilla)

    df = df_base.copy()
    
    df['r'] = np.clip(rng.normal(0, 1, len(df)), -4, 4)
    df['sigma_T'] = df.apply(
        lambda row: calcular_desviacion_clark(row['v0'], row['t'], valor_alpha),
        axis=1
    )
    df['vT'] = df['v0'] + df['sigma_T'] * df['r']
    df['vT'] = df['vT'].clip(lower=0)

    historial = []
    for index, row in df.iterrows():
        n = row['n']
        T = int(row['t'])
        v0 = row['v0']
        vT = row['vT']

        for t in range(T, -1, -1):
            dia_actual = T - t

            if t == 0:
                v_t = v0
                F_t = v0
                r_t = 0
            else:
                factor_T = factor_incertidumbre_no_lineal(T)
                factor_t = factor_incertidumbre_no_lineal(t)
                peso_error = factor_t / factor_T if factor_T > 0 else 0
                v_t = v0 + peso_error * (vT - v0)
                r_t = np.clip(rng.normal(0, 1), -4, 4)
                sigma_t = calcular_desviacion_clark(v_t, t, valor_alpha)
                F_t = max(0, v_t + sigma_t * r_t)  # v_t(1+ raiz(log(1+t)*alpha*r_t)) y v_t*raiz(log(1+t)*alpha*r_t = sigma_t*r_t
            
            historial.append({
                'Semilla': semilla, # <--- Agregamos qué semilla generó esta fila
                'Fecha de Hoy': dia_actual, 'n': n, 'T': T, 't': t, 
                'v0': v0, 'vT': vT, 'v_t': v_t, 'F_t': F_t, 'r_t': r_t
            })
            
    return pd.DataFrame(historial)

semillas_a_probar = [40] # Para el archivo final, una sola semilla es suficiente
lista_dfs = []

for s in semillas_a_probar:
    # --- CORRECCIÓN: Llamar a la simulación UNA SOLA VEZ con los datos extendidos ---
    rng = np.random.default_rng(s)
    df_escenario = simular_escenario(df_demands_extended, alpha, s, rng=rng)
    lista_dfs.append(df_escenario)

# Unimos todos los escenarios en una sola super tabla
df_evolucion_multisemilla = pd.concat(lista_dfs, ignore_index=True)

def graficar_multisemilla(df, mezcla, periodo_T, valor_alpha):
    # Filtramos por mezcla y tiempo de entrega
    df_plot = df[(df['n'] == mezcla) & (df['T'] == periodo_T)].copy()
    
    if df_plot.empty:
        print(f"⚠️ No se encontraron datos para la mezcla '{mezcla}' entregada en T={periodo_T}")
        return
    
    plt.figure(figsize=(12, 7)) # Un poco más ancho para que se vea mejor
    v0_real = df_plot['v0'].iloc[0]
    
    # Obtenemos las semillas únicas y una paleta de colores de matplotlib
    semillas_unicas = df_plot['Semilla'].unique()
    colores = plt.cm.tab10.colors 
    
    # Graficamos cada escenario uno por uno
    for i, semilla in enumerate(semillas_unicas):
        # Filtramos solo los datos de esta semilla y ordenamos el tiempo
        df_s = df_plot[df_plot['Semilla'] == semilla].sort_values('t', ascending=False)
        color_actual = colores[i % len(colores)] # Asignamos un color distinto
        
        # Línea recta base interpolada (v_t) - Punteada y más transparente
        plt.plot(df_s['t'], df_s['v_t'], 
                 color=color_actual, linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Línea del pronóstico con ruido (F_t) - Sólida con marcadores
        plt.plot(df_s['t'], df_s['F_t'], 
                 color=color_actual, marker='.', markersize=6, linestyle='-', 
                 linewidth=1.5, label=f'Escenario (Semilla {semilla})')
    
    # Línea horizontal negra gruesa para la Demanda Real (v0)
    plt.axhline(y=v0_real, color='black', linestyle='-', linewidth=3, 
                label=f'Demanda Real (v0 = {v0_real})')
    
    plt.gca().invert_xaxis()
    
    plt.title(f"Evolución de Pronósticos $F_t$ (Múltiples Semillas)\nMezcla: {mezcla} | T de Entrega: {periodo_T} | Alpha: {valor_alpha}", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Periodos restantes antes de la demanda real (t)", fontsize=12)
    plt.ylabel("Demanda (Unidades)", fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --- 6. EXPORTAR EL ARCHIVO FINAL ---
output_filename = "Evolucion_Pronosticos.xlsx"
if os.path.exists(output_filename):
    os.remove(output_filename) # Borramos el viejo para evitar conflictos
df_evolucion_multisemilla.to_excel(output_filename, index=False, engine='openpyxl')
print(f"Archivo de pronósticos extendido '{output_filename}' generado correctamente.")

def mostrar_datos_consola(df, mezcla='n1', periodo_T=16):
    df_filtrado = df[(df['n'] == mezcla) & (df['T'] == periodo_T)].copy()

    if df_filtrado.empty:
        print(f"No se encontraron datos para mezcla={mezcla}, T={periodo_T}.")
        return

    df_filtrado = df_filtrado.sort_values(['Semilla', 't'], ascending=[True, False])
    columnas = ['Semilla', 'Fecha de Hoy', 'n', 'T', 't', 'v0', 'vT', 'v_t', 'F_t', 'r_t']
    df_filtrado = df_filtrado[columnas]

    columnas_redondeo = ['v0', 'vT', 'v_t', 'F_t', 'r_t']
    df_filtrado[columnas_redondeo] = df_filtrado[columnas_redondeo].round(2)

    print("\n" + "=" * 90)
    print(f"Datos generados para mezcla={mezcla}, T={periodo_T}")
    print("=" * 90)
    print(df_filtrado.to_string(index=False))
mostrar_datos_consola(df_evolucion_multisemilla, mezcla='n1', periodo_T=16)

# --- 5. PRUEBA DEL GRÁFICO ---
# Para n1 entregado en T=16 (Puedes probar con n1 y T=150 también)
# graficar_multisemilla(df_evolucion_multisemilla, mezcla='n1', periodo_T=16, valor_alpha=alpha)
graficar_multisemilla(df_evolucion_multisemilla, mezcla='n1', periodo_T=16, valor_alpha=alpha)


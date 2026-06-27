import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CARGA DE DATOS ---
df_demands = pd.read_excel('Datos_Originales.xlsx', sheet_name='Demands')
df_demands.rename(columns={'n': 'n', 't': 't', 'D': 'v0'}, inplace=True)

# --- EXTENSIÓN ARTIFICIAL DEL HORIZONTE ---
# Para correr más iteraciones, necesitamos pronósticos más lejanos.
# Vamos a duplicar las demandas existentes y empujarlas 210 días hacia el futuro.
df_demands_future = df_demands.copy()
df_demands_future['t'] = df_demands_future['t'] + 210 # Empujamos el tiempo de entrega

# Unimos la demanda original con la futura
df_demands_extended = pd.concat([df_demands, df_demands_future], ignore_index=True)

alpha = 0.05

# --- 2. FUNCIÓN DE SIMULACIÓN ---
# Metemos tu lógica en una función que recibe la semilla
def simular_escenario(df_base, valor_alpha, semilla):
    np.random.seed(semilla)
    df = df_base.copy()
    
    df['r'] = np.random.normal(0, 1, len(df))
    df['vT'] = df['v0'] * (1 + df['t'] * valor_alpha * df['r'])
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
                v_t = v0 + (t/T) * (vT - v0)
                r_t = np.random.normal(0, 1)
                F_t = max(0, v_t * (1 + t * valor_alpha * r_t))
            
            historial.append({
                'Semilla': semilla, # <--- Agregamos qué semilla generó esta fila
                'Fecha de Hoy': dia_actual, 'n': n, 'T': T, 't': t, 
                'v0': v0, 'vT': vT, 'v_t': v_t, 'F_t': F_t, 'r_t': r_t
            })
            
    return pd.DataFrame(historial)

semillas_a_probar = [42] # Para el archivo final, una sola semilla es suficiente
lista_dfs = []

for s in semillas_a_probar:
    # --- CORRECCIÓN: Llamar a la simulación UNA SOLA VEZ con los datos extendidos ---
    df_escenario = simular_escenario(df_demands_extended, alpha, s)
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

# --- 5. PRUEBA DEL GRÁFICO ---
# Para n1 entregado en T=16 (Puedes probar con n1 y T=150 también)
# graficar_multisemilla(df_evolucion_multisemilla, mezcla='n1', periodo_T=16, valor_alpha=alpha)
# graficar_multisemilla(df_evolucion_multisemilla, mezcla='n1', periodo_T=16, valor_alpha=alpha)

# --- 6. EXPORTAR EL ARCHIVO FINAL ---
output_filename = "Evolucion_Pronosticos.xlsx"
if os.path.exists(output_filename):
    os.remove(output_filename) # Borramos el viejo para evitar conflictos
df_evolucion_multisemilla.to_excel(output_filename, index=False, engine='openpyxl')
print(f"Archivo de pronósticos extendido '{output_filename}' generado correctamente.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Forecasting_Clark_2 import df_evolucion

def graficar_evolucion_pronostico(df, mezcla, periodo_T, valor_alpha):
    # Filtrar datos usando tus nombres de columnas
    df_plot = df[(df['n'] == mezcla) & (df['T'] == periodo_T)].copy()
    
    if df_plot.empty:
        print(f"No se encontraron datos para la mezcla '{mezcla}' entregada en T={periodo_T}")
        return
    
    # Ordenar por tiempo restante 't'
    df_plot = df_plot.sort_values('t', ascending=False)
    
    plt.figure(figsize=(10, 6))
    v0_real = df_plot['v0'].iloc[0]
    
    # Línea recta base interpolada (v_t)
    plt.plot(df_plot['t'], df_plot['v_t'], 
             color='orange', linestyle='--', linewidth=2, 
             label='Base Interpolada (v_t)')
    
    # Línea del pronóstico con ruido (F_t)
    plt.plot(df_plot['t'], df_plot['F_t'], 
             color='blue', marker='o', markersize=4, linestyle='-', linewidth=1.5, 
             label='Pronóstico Actualizado (F_t)')
    
    # Línea horizontal de la Demanda Real (v0)
    plt.axhline(y=v0_real, color='green', linestyle='-', linewidth=2, 
                label=f'Demanda Real (v0 = {v0_real})')
    
    # Invertir el eje X (Queremos que vaya desde T bajando hasta 0)
    plt.gca().invert_xaxis()
    
    plt.title(f"Evolución de los Pronósticos $F_t$\nMezcla: {mezcla} | T de Entrega: {periodo_T} | Alpha: {valor_alpha}", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Periodos restantes antes de la demanda real (t)", fontsize=12)
    plt.ylabel("Demanda (Unidades)", fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()

graficar_evolucion_pronostico(df_evolucion, mezcla='n1', periodo_T=210, valor_alpha=0.05)
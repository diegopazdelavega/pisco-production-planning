import pandas as pd
import numpy as np
from Integrar_Forecasting import calcular_desviacion_clark, factor_incertidumbre_no_lineal

df_demands = pd.read_excel('Datos_Originales.xlsx', sheet_name='Demands')
df_demands.rename(columns={'n': 'n', 't': 't', 'D': 'v0'}, inplace=True)

# v0 = df_demands["v0"]
# print(v0)

rng = np.random.default_rng(42)  # Semilla para reproducibilidad
alpha = 0.05

df_demands['r'] = np.clip(rng.normal(0, 1, len(df_demands)), -4, 4)
df_demands['sigma_T'] = df_demands.apply(
    lambda row: calcular_desviacion_clark(row['v0'], row['t'], alpha),
    axis=1
)

df_demands['vT'] = df_demands['v0'] + df_demands['sigma_T'] * df_demands['r']

df_demands['vT'] = df_demands['vT'].clip(lower=0)

#print(df_demands[['n', 't', 'v0', 'r', 'vT']].head(10))
#print(df_demands[['n', 't', 'v0', 'r', 'vT']].tail(10))
#print(df_demands[['n', 't', 'v0', 'r', 'vT']].nunique())
#print(df_demands[['n', 't', 'v0', 'r', 'vT']][df_demands['n'] == 'n1'])

historial_pronosticos = []

for index, row in df_demands.iterrows():
    n = row['n']
    T = int(row['t'])
    v0 = row['v0']
    vT = row['vT']

    for t in range(T, -1, -1):
        dia_actual = T - t

        if t ==0:
            v_t = v0
            F_t = v0
            r_t = 0
        else:
            factor_T = factor_incertidumbre_no_lineal(T)
            factor_t = factor_incertidumbre_no_lineal(t)
            peso_error = factor_t / factor_T if factor_T > 0 else 0
            v_t = v0 + peso_error * (vT - v0)

            r_t = np.clip(rng.normal(0, 1), -4, 4)
            sigma_t = calcular_desviacion_clark(v_t, t, alpha)
            F_t = max(0, v_t + sigma_t * r_t)
        
        historial_pronosticos.append({'Fecha de Hoy': dia_actual, 'n': n, 'T': T, 't': t, 'v0': v0, 'vT': vT, 'v_t': v_t, 'F_t': F_t, 'r_t': r_t})

df_evolucion = pd.DataFrame(historial_pronosticos)

df_n1_16 = df_evolucion[(df_evolucion['n'] == 'n1') & (df_evolucion['T'] == 16)]
#print(df_evolucion[['Fecha de Hoy','n', 'T', 't', 'v0', 'vT', 'v_t', 'F_t','r_t']].head(17))
print(df_n1_16[['Fecha de Hoy','n', 'T', 't', 'v0', 'vT', 'v_t', 'F_t','r_t']])

#df_evolucion[['Fecha de Hoy','n', 'T', 't', 'v0', 'vT', 'v_t', 'F_t','r_t']].to_excel('Evolucion_Pronosticos.xlsx', index=False)

#print(df_evolucion.info())

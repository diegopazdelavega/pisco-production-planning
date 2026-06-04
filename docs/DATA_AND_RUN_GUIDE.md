# Data and Run Guide

This guide describes the workbook schema, run commands, outputs, and practical maintenance notes for the pisco planning models.

## Workbook

The project uses `Datos_Originales.xlsx`. The scripts load every worksheet with `pandas.read_excel(..., sheet_name=None)` and strip whitespace from column names.

## Required Sheets

| Sheet | Required columns | Description |
| --- | --- | --- |
| `Set_T` | `t` | Time index. The value `0` is used for initial-state logic; decisions use `t != 0`. |
| `Set_A` | `a` | Alcohol identifiers. |
| `Set_N` | `n` | Blend identifiers. |
| `Set_P` | `p` | Process identifiers. Current code expects names such as `reception`, `mixing`, `infusion`, `barrel`, `stabilization`, `refinement`. |
| `Demands` | `n`, `t`, `D` | Demand by blend and day. |
| `Recipes` | `n`, `a`, `R` | Fraction of alcohol `a` required by blend `n`. |
| `Routes_Taus` | `n`, `p`, `tau`, `Cwip`, `valid` | Valid process routes, process duration, and process inventory cost. The `valid` column is read but not currently used in the model builder. |
| `Capacities` | `p`, `CAP_p`, `CAP_labor` | Storage capacity and daily labor/throughput capacity by process. |
| `Scalars` | `L`, `Qmin`, `M`, `Cv`, `Ci_minus` | Scalar defaults. These are present in data but mostly hard-coded in Python today. |
| `Inv_Inicial_Alcoholes` | `a`, `Ia` | Initial raw alcohol inventory. |
| `LLegadas_Alcoholes_Omega` | `a`, `wa` | Scheduled initial raw alcohol arrivals. |
| `Backlog_Inicial` | `n`, `I_minus_n` | Initial backlog by blend. Can be empty. |
| `Inventario_Antiguo_Gamma` | `n`, `p`, `u`, `gamma` | Initial WIP with age `u`. Can be empty. |
| `Asignacion_de_Grupos_N(p)` | `Group`, `N(<process>)` columns | Which blends belong to each initial-inventory group by process. |
| `Inv_Inicial_Grupos_N(p)` | `Group`, `N(<process>)` columns | Initial grouped ready inventory by process. |

## Current Workbook Shape

The included workbook currently has:

| Sheet | Rows x columns |
| --- | --- |
| `Set_T` | `91 x 1` |
| `Set_A` | `9 x 1` |
| `Set_N` | `8 x 1` |
| `Set_P` | `6 x 1` |
| `Demands` | `160 x 3` |
| `Recipes` | `23 x 3` |
| `Routes_Taus` | `27 x 5` |
| `Capacities` | `6 x 3` |
| `Scalars` | `1 x 5` |
| `Asignacion_de_Grupos_N(p)` | `8 x 6` |
| `Inv_Inicial_Grupos_N(p)` | `8 x 6` |

## Environment Setup

Install Python dependencies:

```bash
pip install pandas numpy pyomo openpyxl gurobipy
```

Install and license Gurobi for the local machine, then verify Pyomo can reach it:

```bash
python3 - <<'PY'
import pyomo.environ as pyo
print(pyo.SolverFactory("gurobi").available())
PY
```

If this prints `False`, fix the Gurobi installation or Python environment before running the optimization scripts.

The repository does not depend on a project-local `gurobi.lic`, `GRB_LICENSE_FILE`, or `GUROBI_HOME`. Let Gurobi use its standard license discovery process for each workstation:

- install or activate the license for the current machine with Gurobi's normal tooling, such as `grbgetkey` when using a named-user or academic license,
- place any required license file in a standard Gurobi location for the operating system,
- or configure environment variables outside the repository if your organization requires them.

Do not commit machine-specific license files. The repository ignores `gurobi.lic` and `*.lic`.

The scripts also run a startup validation through `solver_utils.py`. This check verifies that Pyomo can create the Gurobi solver and that Gurobi can solve a tiny one-variable model. If the local license is invalid, the script prints a clear configuration error before attempting the full pisco model.

## Running the Models

Run the scripts from the repository root. The `__main__` sections use the relative workbook path `Datos_Originales.xlsx`.

### Deterministic Static Model

```bash
python3 PiscoDet.py
```

Default solver settings:

- solver: `gurobi`
- time limit: 240 seconds
- output: `Resultados_Determinista.xlsx`

### Stochastic Static Model

```bash
python3 PiscoV3_Estc.py
```

Default stochastic settings:

- scenarios: 10
- demand variability: 20 percent
- random seed: 42
- solver: `gurobi`
- time limit: 240 seconds
- MIP gap: 0.001
- output: `Resultados_Estocásticos.xlsx`

### Deterministic Rolling Horizon

```bash
python3 Pisco_Rolling_Horizon.py
```

Default rolling settings:

- frozen/update period `RP`: 30 days
- nervousness penalty after first iteration: 5.0
- iterations in `__main__`: 4
- solver time limit: 900 seconds
- MIP gap: 0.06
- output per iteration: `Resultados_RH_Det_Iteracion_<i>.xlsx`

### Stochastic Rolling Horizon

```bash
python3 Pisco_Rolling_Horizon_Estc.py
```

Default rolling settings:

- initial SAA scenarios: 10
- demand variability: 20 percent
- frozen/update period `RP`: 30 days
- nervousness penalty after first iteration: 5.0
- iterations in `__main__`: 8
- solver time limit: 1800 seconds
- MIP gap: 0.06
- output per iteration: `Resultados_RH_Periodo_<i>.xlsx`

## Result Workbook Sheets

All model variants call an `ExportarResultados` method that writes a subset of variables to Excel.

| Sheet | Contents |
| --- | --- |
| `y_produccion` | Positive blend production `y[n,t]`. |
| `v_inv_alcohol` | Raw alcohol stock `v[a,t]`. |
| `k_stock_listo` | Positive ready stock `k`. Stochastic outputs include scenario. |
| `i_backlog` | Positive backlog `i_minus`. Stochastic outputs include scenario. |
| `q_recepcion` | Positive alcohol receipts `q[a,t]`. |
| `z_en_proceso` | Positive WIP by process age `z[n,p,t,u]`. |

If a result table is empty, the exporter still writes the sheet with headers.

## Common Failure Modes

### `FileNotFoundError` or no data loaded

The `ReadExcelFile` method silently returns on missing files. If a later line fails with missing attributes such as `Tiempo` or `Demands`, check the workbook path first.

### Gurobi unavailable

The code creates Gurobi through `solver_utils.create_gurobi_solver()`. If Gurobi is not installed, Pyomo cannot solve the MILP.

### Gurobi license invalid

If Gurobi is installed but the license belongs to another machine, the startup check reports the local license problem. A typical symptom is:

```text
HostID mismatch
```

Install or activate a valid license for the current machine. Do not fix this by copying a `gurobi.lic` from another workstation into the project folder.

### Excel output permission error

If the output workbook is open in Excel, `ExportarResultados` catches `PermissionError` and prints an error. Close the workbook and rerun.

### Workbook route edits do not change flows

The next-process topology is hard-coded in `get_next_process`. Update that function when adding route patterns, not only the `Routes_Taus` sheet.

### Scalar sheet changes have no effect

The `Scalars` sheet is loaded, but the model uses hard-coded values for `L`, `Q_min`, `Cv`, `Ci_minus`, and `M`. Update the Python constants or refactor the code to read `Scalars`.

## Recommended Next Engineering Tasks

1. Centralize the duplicated deterministic and stochastic formulation code.
2. Read scalar values from `Scalars` consistently.
3. Add a small smoke test that loads `Datos_Originales.xlsx` and builds each Pyomo instance without solving.
4. Add solver configuration through function arguments or a config file.
5. Validate workbook schemas before building the model.

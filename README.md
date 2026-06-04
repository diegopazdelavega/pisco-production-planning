# Pisco Production Planning

Foundational optimization models for production planning in Chilean pisco operations. The project models alcohol reception, blend preparation, maturation through process routes, finished stock, capacity usage, and backlog penalties using Pyomo mixed-integer linear programming.

## Repository Contents

| File | Purpose |
| --- | --- |
| `Datos_Originales.xlsx` | Source workbook with sets, demand, recipes, routes, capacities, scalar parameters, and initial conditions. |
| `PiscoDet.py` | Deterministic operational planning model. |
| `PiscoV3_Estc.py` | Stochastic Sample Average Approximation (SAA) model with demand scenarios. |
| `Pisco_Rolling_Horizon.py` | Deterministic rolling horizon wrapper with nervousness penalties. |
| `Pisco_Rolling_Horizon_Estc.py` | Stochastic rolling horizon wrapper with scenario updates. |
| `solver_utils.py` | Portable Gurobi startup validation and solver creation helper. |
| `docs/MODEL_FORMULATION.md` | Mathematical and implementation-level model description. |
| `docs/DATA_AND_RUN_GUIDE.md` | Workbook schema, outputs, and execution guidance. |

## Business Problem

The model plans production over a finite daily horizon for multiple pisco blends. It decides:

- how much raw alcohol to receive by alcohol type,
- how much of each blend to produce,
- how material flows through mixing, infusion, barrel, stabilization, and refinement,
- how much inventory remains in process or ready at each stage,
- how much customer demand is delayed as backlog.

The objective minimizes operating and service costs: raw alcohol inventory, work-in-process inventory, finished stock, and backlog. The stochastic version minimizes expected cost over sampled demand scenarios.

## Model Family

The codebase contains four related model variants:

| Variant | Entry point | Demand treatment | Planning mode |
| --- | --- | --- | --- |
| Deterministic | `PiscoDet.py` | Uses workbook demand `D` directly. | Single static horizon. |
| Stochastic SAA | `PiscoV3_Estc.py` | Generates normal demand samples around workbook demand. | Single static horizon. |
| Deterministic rolling horizon | `Pisco_Rolling_Horizon.py` | Re-reads shifted deterministic demand as the calendar advances. | Re-solves and updates initial state after each frozen period. |
| Stochastic rolling horizon | `Pisco_Rolling_Horizon_Estc.py` | Rebuilds SAA demand scenarios after each calendar shift. | Re-solves, samples a realized scenario, and updates state. |

## Current Data Snapshot

The included workbook currently defines:

- 91 time rows: `t = 0..90`, with optimization decisions on `t = 1..90`.
- 9 alcohol types: `a1..a9`.
- 8 blends: `n1..n8`.
- 6 processes: `reception`, `mixing`, `infusion`, `barrel`, `stabilization`, `refinement`.
- 160 demand records, 23 recipe records, and 27 route/process records.
- Scalar defaults: truck lot `L = 30000`, minimum process lot `Qmin = 10000`, big-M `M = 100000`, alcohol inventory cost `Cv = 20`, backlog penalty `Ci_minus = 100`.

## Requirements

Python packages:

```bash
pip install pandas numpy pyomo openpyxl
```

Solver:

- The scripts use `solver_utils.create_gurobi_solver`, which calls Pyomo's `SolverFactory("gurobi")`.
- A working local Gurobi installation and license are required unless the code is adapted to another MILP solver supported by Pyomo.
- Do not commit or depend on a project-local `gurobi.lic`; Gurobi should discover the license through its standard local installation process.

## Quick Start

Run one model from the repository root:

```bash
python3 PiscoDet.py
```

For a more controlled run without editing the file, import the class:

```python
from PiscoDet import PiscoModel_Operativo

model = PiscoModel_Operativo()
model.ReadExcelFile("Datos_Originales.xlsx")
instance = model.Solver()
```

## Outputs

Successful solves export Excel result files. Depending on the model, the default output names include:

- `Resultados_Determinista.xlsx`
- `Resultados_Estocásticos.xlsx`
- `Resultados_RH_Det_Iteracion_<i>.xlsx`
- `Resultados_RH_Periodo_<i>.xlsx`

Each result workbook contains sheets for production, alcohol inventory, ready stock, backlog, alcohol reception, and work-in-process aging.

## Important Implementation Notes

- `PiscoDet.py` and `PiscoV3_Estc.py` duplicate most of the base formulation. Future maintenance would be easier if the shared model builder were extracted into a common module.
- Solver startup is centralized in `solver_utils.py`. It verifies that Gurobi is available and runs a tiny validation solve to detect local license errors before the full model is solved.
- Scalar values are present in `Datos_Originales.xlsx`, but the model currently hard-codes them in Python. Changes to the `Scalars` sheet will not automatically affect all scripts.
- The process topology is hard-coded in `get_next_process`; adding blends or route types may require code changes even if workbook routes are updated.
- The stochastic model uses `np.random.seed(42)` for reproducible initial SAA scenarios.
- Rolling horizon wrappers mutate Pyomo parameters in place after each iteration to update initial inventory, backlog, WIP, group stock, reference plans, and demand.

## Documentation

Start with:

- [Model Formulation](docs/MODEL_FORMULATION.md)
- [Data and Run Guide](docs/DATA_AND_RUN_GUIDE.md)

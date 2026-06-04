import functools

import pyomo.environ as pyo


class GurobiConfigurationError(RuntimeError):
    """Raised when Gurobi is installed but cannot be used locally."""


def _format_gurobi_error(exc):
    message = str(exc)
    lower = message.lower()

    if "hostid mismatch" in lower:
        reason = (
            "Gurobi found a license, but it belongs to another machine "
            "(HostID mismatch)."
        )
    elif "license" in lower:
        reason = "Gurobi reported a license error."
    else:
        reason = "Gurobi could not solve a tiny validation model."

    return (
        f"{reason}\n\n"
        "Install or activate a valid local Gurobi license using the standard "
        "Gurobi discovery process for this machine. Do not rely on a "
        "project-specific gurobi.lic file or a hard-coded license path.\n\n"
        f"Original solver error:\n{message}"
    )


@functools.lru_cache(maxsize=1)
def validate_gurobi_ready():
    """Verify that Pyomo can reach Gurobi and that the local license works."""
    solver = pyo.SolverFactory("gurobi")
    if not solver.available(exception_flag=False):
        raise GurobiConfigurationError(
            "Gurobi is not available through Pyomo. Install Gurobi and "
            "gurobipy, then ensure the local environment can discover them."
        )

    try:
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addVar(lb=0.0, name="x")
                model.setObjective(x, gp.GRB.MINIMIZE)
                model.addConstr(x >= 1.0)
                model.optimize()
                if model.Status != gp.GRB.OPTIMAL:
                    raise GurobiConfigurationError(
                        "Gurobi is available, but its startup validation solve "
                        f"did not finish optimally. Status: {model.Status}."
                    )
    except Exception as exc:
        if isinstance(exc, GurobiConfigurationError):
            raise
        raise GurobiConfigurationError(_format_gurobi_error(exc)) from exc

    return True


def create_gurobi_solver(options=None):
    """Create a portable Gurobi solver after validating local availability."""
    validate_gurobi_ready()
    solver = pyo.SolverFactory("gurobi")
    for key, value in (options or {}).items():
        solver.options[key] = value
    return solver

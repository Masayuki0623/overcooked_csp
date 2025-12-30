from typing import Optional, Dict, Any
from ortools.sat.python import cp_model

class SolveResult:
    def __init__(self, status_name: str, status: int, objective_value: Optional[float], solution: Dict[str, int]):
        self.status_name = status_name
        self.status = status
        self.objective_value = objective_value
        self.solution = solution


def solve(csp_model, time_limit: Optional[float] = None, num_workers: Optional[int] = None) -> SolveResult:
    """
    CSPModel を CP-SAT で解く。
    戻り値: SolveResult(status_name, status, objective_value, solution)
    """
    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    if num_workers is not None:
        solver.parameters.num_search_workers = int(num_workers)

    status = solver.Solve(csp_model.model)
    status_name = solver.StatusName(status)
    objective_value = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        try:
            objective_value = solver.ObjectiveValue()
        except Exception:
            objective_value = None

    solution: Dict[str, int] = {}
    for name, var in csp_model.vars.items():
        solution[name] = solver.Value(var)

    return SolveResult(status_name, status, objective_value, solution)

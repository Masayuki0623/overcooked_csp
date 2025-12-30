from .model import CSPModel
from .solver import solve

# 簡易デモ: 予算10で利益最大化
items = [
    ("taskA", 4, 5),  # (name, duration, weight)
    ("taskB", 6, 7),
    ("taskC", 3, 3),
]

budget = 10

m = CSPModel()
var_names = []
durations = {}
benefits = {}
for i, (name, dur, w) in enumerate(items):
    vname = f"x_{name}_{i}"
    m.add_bool_var(vname)
    var_names.append(vname)
    durations[vname] = dur
    benefits[vname] = w * dur

m.add_linear_le(durations, budget)
m.maximize_linear(benefits)

res = solve(m, time_limit=2.0)
print("status:", res.status_name, "obj:", res.objective_value)
selected = [n for n in var_names if res.solution.get(n, 0) == 1]
print("selected:", selected)

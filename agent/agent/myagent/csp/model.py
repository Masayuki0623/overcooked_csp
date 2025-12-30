from typing import Dict, List, Tuple, Iterable, Optional
from ortools.sat.python import cp_model

class CSPModel:
    """
    OR-Tools CP-SAT 用の薄いラッパー。
    - 変数（Int/Bool）を名前で管理
    - ドメイン（範囲 or 離散集合）を簡便に指定
    - 代表的な制約/目的関数を提供
    """
    def __init__(self) -> None:
        self.model = cp_model.CpModel()
        self.vars: Dict[str, cp_model.IntVar] = {}

    # --- 変数定義 ---
    def add_int_var(self, name: str, domain: Iterable[int] | Tuple[int, int]) -> cp_model.IntVar:
        """
        IntVar を追加。
        - domain が (lb, ub) のタプルなら区間ドメイン
        - domain がリストやセットなら離散ドメイン（AllowedAssignments）で制限
        """
        if isinstance(domain, tuple):
            lb, ub = domain
            var = self.model.NewIntVar(lb, ub, name)
        else:
            values = sorted(set(domain))
            lb, ub = min(values), max(values)
            var = self.model.NewIntVar(lb, ub, name)
            # 離散値に制限
            self.model.AddAllowedAssignments([var], [[v] for v in values])
        self.vars[name] = var
        return var

    def add_bool_var(self, name: str) -> cp_model.IntVar:
        var = self.model.NewBoolVar(name)
        self.vars[name] = var
        return var

    # --- 制約 ---
    def add_all_different(self, names: List[str]) -> None:
        self.model.AddAllDifferent([self.vars[n] for n in names])

    def add_equal(self, a: str, b: str) -> None:
        self.model.Add(self.vars[a] == self.vars[b])

    def add_not_equal(self, a: str, b: str) -> None:
        self.model.Add(self.vars[a] != self.vars[b])

    def add_less_than(self, a: str, b: str) -> None:
        self.model.Add(self.vars[a] < self.vars[b])

    def add_sum_equals(self, names: List[str], value: int) -> None:
        self.model.Add(sum(self.vars[n] for n in names) == value)

    def add_linear_le(self, name_coeffs: Dict[str, int], rhs: int) -> None:
        expr = sum(self.vars[n] * c for n, c in name_coeffs.items())
        self.model.Add(expr <= rhs)

    def add_linear_ge(self, name_coeffs: Dict[str, int], rhs: int) -> None:
        expr = sum(self.vars[n] * c for n, c in name_coeffs.items())
        self.model.Add(expr >= rhs)

    def add_allowed_assignments(self, names: List[str], tuples: List[List[int]]) -> None:
        self.model.AddAllowedAssignments([self.vars[n] for n in names], tuples)

    # --- 目的関数 ---
    def maximize_linear(self, name_coeffs: Dict[str, int]) -> None:
        expr = sum(self.vars[n] * c for n, c in name_coeffs.items())
        self.model.Maximize(expr)

    def minimize_linear(self, name_coeffs: Dict[str, int]) -> None:
        expr = sum(self.vars[n] * c for n, c in name_coeffs.items())
        self.model.Minimize(expr)

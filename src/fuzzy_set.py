from typing import Dict

import numpy as np

from src.fuzzy_mf import TrapezoidalMF, TriangularMF

MF_TYPES = {"triangular": TriangularMF, "trapezoidal": TrapezoidalMF}


class FuzzySet:
    def __init__(self, name: str, mf):
        self.name = name
        self.mf = mf

    def get_membership(self, x):
        return self.mf.evaluate(x)


class LinguisticVariable:
    def __init__(self, name: str, min_val: float, max_val: float, unit: str = ""):
        self.name, self.min_val, self.max_val, self.unit = name, min_val, max_val, unit
        self.terms: Dict[str, FuzzySet] = {}

    def add_term(self, name: str, mf) -> None:
        self.terms[name] = FuzzySet(name, mf)

    def fuzzify(self, x: float) -> Dict[str, float]:
        x = float(np.clip(x, self.min_val, self.max_val))
        return {n: float(t.get_membership(x)) for n, t in self.terms.items()}

    def universe(self, n_points: int = 601) -> np.ndarray:
        return np.linspace(self.min_val, self.max_val, n_points)

    def min_coverage(self, n_points: int = 2001) -> float:
        xs = self.universe(n_points)
        return float(np.max([t.get_membership(xs) for t in self.terms.values()], axis=0).min())

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "LinguisticVariable":
        lo, hi = cfg["domain"]
        var = cls(name, lo, hi, cfg.get("unit", ""))
        for term, spec in cfg["terms"].items():
            var.add_term(term, MF_TYPES[spec["type"]](*spec["params"]))
        return var

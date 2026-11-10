"""Двигуни нечіткого виводу: Мамдані (min-max-centroid) та Такагі-Сугено 1-го порядку."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.fuzzy_rules import RuleBase
from src.fuzzy_set import LinguisticVariable

EPS = 1e-9


@dataclass
class InferenceResult:
    output: float
    active_rules: int
    strengths: List[float]
    aggregated: Optional[np.ndarray] = field(default=None, repr=False)   # лише для Мамдані


class _BaseEngine:
    def __init__(self, inputs: Dict[str, LinguisticVariable], output: LinguisticVariable, rulebase: RuleBase):
        self.inputs, self.output, self.rulebase = inputs, output, rulebase

    def _fuzzify(self, crisp: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return {n: var.fuzzify(crisp[n]) for n, var in self.inputs.items()}


class MamdaniEngine(_BaseEngine):
    def __init__(self, inputs, output, rulebase, n_points: int = 601, tnorm: str = "min"):
        super().__init__(inputs, output, rulebase)
        self.tnorm = tnorm
        self.y = output.universe(n_points)
        self._term_mu = {t: s.get_membership(self.y) for t, s in output.terms.items()}

    def infer(self, crisp: Dict[str, float]) -> InferenceResult:
        w = self.rulebase.firing_strengths(self._fuzzify(crisp), self.tnorm)
        agg = np.zeros_like(self.y)
        for wi, rule in zip(w, self.rulebase.rules):
            if wi > EPS:
                np.maximum(agg, np.minimum(wi, self._term_mu[rule.consequent]), out=agg)
        area = agg.sum()
        y_star = float((self.y * agg).sum() / area) if area > EPS else float(self.y.mean())
        return InferenceResult(y_star, sum(wi > EPS for wi in w), w, agg)


class SugenoEngine(_BaseEngine):
    def __init__(self, inputs, output, rulebase, tnorm: str = "prod"):
        super().__init__(inputs, output, rulebase)
        self.tnorm = tnorm
        self._names = list(inputs)

    def infer(self, crisp: Dict[str, float]) -> InferenceResult:
        w = self.rulebase.firing_strengths(self._fuzzify(crisp), self.tnorm)
        x = [float(np.clip(crisp[n], self.inputs[n].min_val, self.inputs[n].max_val)) for n in self._names]
        num = den = 0.0
        for wi, rule in zip(w, self.rulebase.rules):
            if wi > EPS:
                a, b, c = rule.sugeno
                num += wi * (a * x[0] + b * x[1] + c)
                den += wi
        y = num / den if den > EPS else float(self.output.universe(2).mean())
        y = float(np.clip(y, self.output.min_val, self.output.max_val))
        return InferenceResult(y, sum(wi > EPS for wi in w), w)

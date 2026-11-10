from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.fuzzy_set import LinguisticVariable

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "rules_config.yaml"


@dataclass(frozen=True)
class Rule:
    antecedent: Tuple[Tuple[str, str], ...]          # (("Energy","Low"), ("EventRate","High"))
    consequent: str                                   # терм виходу (Мамдані)
    sugeno: Optional[Tuple[float, float, float]] = None   # (a, b, c) для TSK

    def __str__(self) -> str:
        cond = " AND ".join(f"{v} is {t}" for v, t in self.antecedent)
        return f"IF {cond} THEN SleepTime is {self.consequent}"


class RuleBase:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def __len__(self) -> int:
        return len(self.rules)

    def firing_strengths(self, memberships: Dict[str, Dict[str, float]], tnorm: str = "min") -> List[float]:
        out = []
        for r in self.rules:
            mus = [memberships[v][t] for v, t in r.antecedent]
            w = min(mus) if tnorm == "min" else _prod(mus)
            out.append(w)
        return out

    def check_completeness(self, inputs: Dict[str, LinguisticVariable]) -> List[str]:
        present = {tuple(r.antecedent) for r in self.rules}
        names = list(inputs)
        missing = []
        for combo in product(*(list(inputs[n].terms) for n in names)):
            if tuple(zip(names, combo)) not in present:
                missing.append(" AND ".join(f"{n}={t}" for n, t in zip(names, combo)))
        return missing

    def check_consistency(self) -> List[str]:
        seen: Dict[tuple, str] = {}
        conflicts = []
        for r in self.rules:
            key = tuple(r.antecedent)
            if key in seen and seen[key] != r.consequent:
                conflicts.append(f"{key}: {seen[key]} vs {r.consequent}")
            seen[key] = r.consequent
        return conflicts


def _prod(values) -> float:
    p = 1.0
    for v in values:
        p *= v
    return p


def load_system(path: Path = DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inputs = {n: LinguisticVariable.from_config(n, c) for n, c in cfg["inputs"].items()}
    (out_name, out_cfg), = cfg["output"].items()
    output = LinguisticVariable.from_config(out_name, out_cfg)
    rules = []
    for item in cfg["rules"]:
        ant = tuple((v, item["if"][v]) for v in inputs)
        s = item.get("sugeno")
        rules.append(Rule(ant, item["then"], (s["a"], s["b"], s["c"]) if s else None))
    return cfg, inputs, output, RuleBase(rules)

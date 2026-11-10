from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class NodeModel:
    capacity_mah: float = 1000.0
    i_wake_ma: float = 15.0
    t_wake_s: float = 0.05
    t_event_s: float = 0.01
    i_sleep_ma: float = 0.01
    step_min: float = 10.0
    max_days: float = 2000.0


def event_profile(n_steps: int, step_min: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_days = np.arange(n_steps) * step_min / 1440.0
    rate = 15.0 + 12.0 * np.sin(2 * np.pi * t_days) + rng.normal(0.0, 3.0, n_steps)
    return np.clip(rate, 0.0, 50.0)


def simulate(period_fn: Callable[[float, float], float], model: NodeModel = NodeModel(), seed: int = 42) -> Dict:
    n_steps = int(model.max_days * 1440 / model.step_min)
    rates = event_profile(n_steps, model.step_min, seed)
    cap_mas = model.capacity_mah * 3600.0
    remaining = cap_mas
    energy_hist, period_hist = [], []
    delay_sum = events_sum = 0.0
    step_s = model.step_min * 60.0
    steps_done = 0
    for k in range(n_steps):
        e_pct = 100.0 * remaining / cap_mas
        r = rates[k]
        t_sleep = period_fn(e_pct, r)
        wakes = step_s / (t_sleep + model.t_wake_s)
        events = r * model.step_min
        used = (wakes * model.i_wake_ma * model.t_wake_s
                + events * model.i_wake_ma * model.t_event_s
                + model.i_sleep_ma * step_s)
        remaining -= used
        energy_hist.append(e_pct)
        period_hist.append(t_sleep)
        delay_sum += events * t_sleep / 2.0
        events_sum += events
        steps_done = k + 1
        if remaining <= 0:
            break
    return {
        "lifetime_days": steps_done * model.step_min / 1440.0,
        "avg_latency_s": delay_sum / events_sum if events_sum else float("nan"),
        "avg_period_s": float(np.mean(period_hist)),
        "energy": np.array(energy_hist),
        "period": np.array(period_hist),
        "step_min": model.step_min,
    }

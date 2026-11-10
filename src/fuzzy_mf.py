import numpy as np


class TrapezoidalMF:
    def __init__(self, a: float, b: float, c: float, d: float):
        if not a <= b <= c <= d:
            raise ValueError(f"Потрібно a<=b<=c<=d, отримано {(a, b, c, d)}")
        self.a, self.b, self.c, self.d = a, b, c, d

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if self.b > self.a:
            rise = np.clip((x - self.a) / (self.b - self.a), 0.0, 1.0)
        else:
            rise = (x >= self.b).astype(float)
        if self.d > self.c:
            fall = np.clip((self.d - x) / (self.d - self.c), 0.0, 1.0)
        else:
            fall = (x <= self.c).astype(float)
        mu = np.minimum(rise, fall)
        return float(mu) if mu.ndim == 0 else mu


class TriangularMF(TrapezoidalMF):
    def __init__(self, a: float, b: float, c: float):
        super().__init__(a, b, b, c)

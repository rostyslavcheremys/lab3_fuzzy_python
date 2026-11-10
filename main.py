import os
import statistics
import time
import tracemalloc

import numpy as np
import pandas as pd
from tabulate import tabulate

from src import plots
from src.fuzzy_engine import MamdaniEngine, SugenoEngine
from src.fuzzy_rules import load_system
from src.simulation import NodeModel, simulate

REPORTS = "reports"
TEST_CASES = [
    ("Повна батарея, багато подій", {"Energy": 90.0, "EventRate": 40.0}),
    ("Низький заряд, спокій", {"Energy": 15.0, "EventRate": 3.0}),
    ("Середній стан", {"Energy": 60.0, "EventRate": 12.0}),
    ("Перехідна зона (перекриття термів)", {"Energy": 72.0, "EventRate": 28.0}),
    ("Конфлікт: низький заряд, багато подій", {"Energy": 20.0, "EventRate": 30.0}),
]


def measure(engine, crisp, repeats: int = 300):
    engine.infer(crisp)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = engine.infer(crisp)
        times.append((time.perf_counter() - t0) * 1000.0)
    tracemalloc.start()
    engine.infer(crisp)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, statistics.median(times), peak / 1024.0


def main():
    os.makedirs(REPORTS, exist_ok=True)
    print("=" * 96)
    print(" ЛАБОРАТОРНА РОБОТА №3: НЕЧІТКА ЛОГІКА ТА НЕЧІТКИЙ ВИВІД (PYTHON 3.10+)")
    print(" ВАРІАНТ 13: Нечіткий контролер енергозбереження вузла WSN сенсора")
    print("=" * 96 + "\n")

    cfg, inputs, output, rb = load_system()
    mam = MamdaniEngine(inputs, output, rb, n_points=cfg["inference"]["mamdani"]["n_points"])
    sug = SugenoEngine(inputs, output, rb)

    missing, conflicts = rb.check_completeness(inputs), rb.check_consistency()
    print(f"База правил: {len(rb)} правил; повнота: {'так' if not missing else 'НІ ' + str(missing)}; "
          f"суперечності: {'немає' if not conflicts else conflicts}")
    for v in list(inputs.values()) + [output]:
        print(f"  покриття {v.name}: мін. max-належність = {v.min_coverage():.3f}")
    print()

    rows = []
    for title, crisp in TEST_CASES:
        print(f"Тест-кейс «{title}»: Energy = {crisp['Energy']} %, EventRate = {crisp['EventRate']} под./хв")
        case_rows = []
        for name, eng in (("Mamdani", mam), ("Sugeno (TSK)", sug)):
            res, t_ms, mem_kb = measure(eng, crisp)
            case_rows.append({"Алгоритм": name, "SleepTime (с)": f"{res.output:.2f}",
                              "Активні правила": res.active_rules,
                              "Час виводу (мс)": f"{t_ms:.4f}", "Пам'ять (КБ)": f"{mem_kb:.1f}"})
            rows.append({"Кейс": title, **case_rows[-1]})
        print(tabulate(pd.DataFrame(case_rows), headers="keys", tablefmt="github", showindex=False), "\n")
    pd.DataFrame(rows).to_csv(f"{REPORTS}/results.csv", index=False)

    E = np.linspace(0, 100, 101)
    R = np.linspace(0, 50, 51)
    Zm = np.array([[mam.infer({"Energy": e, "EventRate": r}).output for r in R] for e in E])
    Zs = np.array([[sug.infer({"Energy": e, "EventRate": r}).output for r in R] for e in E])
    diff = Zs - Zm

    def smooth(Z):
        gE, gR = np.gradient(Z, E, R)
        g = np.hypot(gE, gR)
        return g.mean(), g.max()

    sm, ss = smooth(Zm), smooth(Zs)
    print("Порівняння поверхонь керування (сітка 101×51):")
    print(f"  Мамдані: діапазон {Zm.min():.2f}..{Zm.max():.2f} с; |grad| середній {sm[0]:.3f}, макс {sm[1]:.3f}")
    print(f"  Сугено : діапазон {Zs.min():.2f}..{Zs.max():.2f} с; |grad| середній {ss[0]:.3f}, макс {ss[1]:.3f}")
    print(f"  Розбіжність: MAE = {np.abs(diff).mean():.2f} с, RMSE = {np.sqrt((diff**2).mean()):.2f} с, "
          f"макс = {np.abs(diff).max():.2f} с\n")

    pts = [{"Energy": float(e), "EventRate": float(r)} for e in E[::5] for r in R[::5]]
    bench = {}
    for name, eng in (("Mamdani", mam), ("Sugeno (TSK)", sug)):
        t0 = time.perf_counter()
        for _ in range(10):
            for p in pts:
                eng.infer(p)
        bench[name] = (time.perf_counter() - t0) * 1000.0 / (10 * len(pts))
        print(f"Середній час виводу {name}: {bench[name]:.4f} мс/виклик ({1000/bench[name]:.0f} викл./с)")

    t0 = time.perf_counter()
    for _ in range(10):
        for p in pts:
            mam._fuzzify(p)
    t_fuzz = (time.perf_counter() - t0) * 1000.0 / (10 * len(pts))
    print(f"З них спільна фазифікація входів: {t_fuzz:.4f} мс; специфічна частина: "
          f"Mamdani ≈ {bench['Mamdani'] - t_fuzz:.4f} мс, Sugeno ≈ {bench['Sugeno (TSK)'] - t_fuzz:.4f} мс")
    print()

    model = NodeModel()
    sims = {
        "Фіксований сон 5 с": simulate(lambda e, r: 5.0, model),
        "Фіксований сон 10 с": simulate(lambda e, r: 10.0, model),
        "Fuzzy Mamdani": simulate(lambda e, r: mam.infer({"Energy": e, "EventRate": r}).output, model),
        "Fuzzy Sugeno": simulate(lambda e, r: sug.infer({"Energy": e, "EventRate": r}).output, model),
    }
    base = sims["Фіксований сон 5 с"]["lifetime_days"]
    sim_rows = [{"Режим": k, "Ресурс (діб)": f"{v['lifetime_days']:.0f}",
                 "Відносно 5 с": f"×{v['lifetime_days']/base:.2f}",
                 "Середній період (с)": f"{v['avg_period_s']:.1f}",
                 "Середня затримка події (с)": f"{v['avg_latency_s']:.1f}"} for k, v in sims.items()]
    print("Симуляція ресурсу батареї (ілюстративна модель вузла, 1000 мА·год):")
    print(tabulate(pd.DataFrame(sim_rows), headers="keys", tablefmt="github", showindex=False))
    pd.DataFrame(sim_rows).to_csv(f"{REPORTS}/battery_simulation.csv", index=False)

    case = TEST_CASES[3][1]
    plots.plot_membership(inputs, output, f"{REPORTS}/membership_functions.png")
    plots.plot_surfaces(E, R, Zm, Zs, f"{REPORTS}/control_surface.png")
    plots.plot_difference(E, R, Zm, Zs, f"{REPORTS}/surface_difference.png")
    plots.plot_sections(mam, sug, f"{REPORTS}/cross_sections.png")
    plots.plot_mamdani_case(mam, case, mam.infer(case), f"{REPORTS}/mamdani_inference.png")
    plots.plot_simulation(sims, f"{REPORTS}/battery_simulation.png")
    print(f"\n[УСПІШНО] Моделювання завершено. Графіки та CSV збережено у {REPORTS}/")


if __name__ == "__main__":
    main()

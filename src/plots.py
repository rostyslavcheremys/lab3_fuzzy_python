import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_membership(inputs, output, path):
    variables = list(inputs.values()) + [output]
    fig, axes = plt.subplots(1, len(variables), figsize=(14, 3.8))
    for ax, var in zip(axes, variables):
        xs = var.universe(1001)
        for name, term in var.terms.items():
            ax.plot(xs, term.get_membership(xs), lw=2, label=name)
        ax.set_title(f"{var.name}, {var.unit}")
        ax.set_xlabel(var.name); ax.set_ylabel("μ"); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_surfaces(E, R, Zm, Zs, path):
    fig = plt.figure(figsize=(13, 5.2))
    EE, RR = np.meshgrid(E, R, indexing="ij")
    for i, (Z, title) in enumerate([(Zm, "Мамдані (центроїд)"), (Zs, "Сугено (TSK, 1-й порядок)")], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_surface(EE, RR, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("Energy, %"); ax.set_ylabel("EventRate, под./хв"); ax.set_zlabel("SleepTime, с")
        ax.set_title(title); ax.view_init(elev=28, azim=-125)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_difference(E, R, Zm, Zs, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ext = [R[0], R[-1], E[0], E[-1]]
    for ax, Z, title, cmap in [(axes[0], Zm, "Мамдані, с", "viridis"), (axes[1], Zs, "Сугено, с", "viridis"),
                               (axes[2], Zs - Zm, "Сугено − Мамдані, с", "coolwarm")]:
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=ext, cmap=cmap)
        ax.set_title(title); ax.set_xlabel("EventRate, под./хв"); ax.set_ylabel("Energy, %")
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_sections(mam, sug, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    r = np.linspace(0, 50, 201)
    for e, c in zip((15, 50, 85), ("C0", "C1", "C2")):
        axes[0].plot(r, [mam.infer({"Energy": e, "EventRate": x}).output for x in r], c, label=f"Мамдані, E={e}%")
        axes[0].plot(r, [sug.infer({"Energy": e, "EventRate": x}).output for x in r], c + "--", label=f"Сугено, E={e}%")
    axes[0].set_xlabel("EventRate, под./хв"); axes[0].set_ylabel("SleepTime, с"); axes[0].legend(fontsize=7)
    e = np.linspace(0, 100, 201)
    for rr, c in zip((5, 20, 40), ("C0", "C1", "C2")):
        axes[1].plot(e, [mam.infer({"Energy": x, "EventRate": rr}).output for x in e], c, label=f"Мамдані, R={rr}")
        axes[1].plot(e, [sug.infer({"Energy": x, "EventRate": rr}).output for x in e], c + "--", label=f"Сугено, R={rr}")
    axes[1].set_xlabel("Energy, %"); axes[1].set_ylabel("SleepTime, с"); axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle("Перерізи поверхні керування (суцільна — Мамдані, пунктир — Сугено)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_mamdani_case(engine, crisp, res, path):
    y = engine.y
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for name, mu in engine._term_mu.items():
        ax.plot(y, mu, color="gray", lw=0.8, alpha=0.6)
    ax.fill_between(y, res.aggregated, color="C0", alpha=0.45, label="Агрегований вихід (max зрізаних термів)")
    ax.axvline(res.output, color="r", lw=2, label=f"Центроїд = {res.output:.2f} с")
    ax.set_xlabel("SleepTime, с"); ax.set_ylabel("μ")
    ax.set_title(f"Вивід Мамдані: Energy={crisp['Energy']}%, EventRate={crisp['EventRate']} под./хв")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_simulation(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for name, r in results.items():
        t = np.arange(len(r["energy"])) * r["step_min"] / 1440.0
        axes[0].plot(t, r["energy"], label=f'{name} ({r["lifetime_days"]:.0f} діб)')
    axes[0].set_xlabel("Час, діб"); axes[0].set_ylabel("Залишок енергії, %"); axes[0].legend(); axes[0].grid(alpha=0.3)
    for name, r in results.items():
        t = np.arange(len(r["period"])) * r["step_min"] / 1440.0
        axes[1].plot(t, r["period"], lw=0.7, label=name)
    axes[1].set_xlim(0, 10); axes[1].set_xlabel("Час, діб (перші 10)"); axes[1].set_ylabel("Період сну, с")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

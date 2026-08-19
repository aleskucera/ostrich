"""Regenerate the comparison figures + summary tables from the result JSONs.

    .venv/bin/python experiments/7_engine_comparison/plot_results.py

Reads results/sweep_*.json, results/speed_dt.json, results/scenarios.json —
whatever exists — and writes results/summary.md plus PNG figures. No numbers are
hand-copied anywhere: this script is the single source for the README tables.
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

R = common.RESULTS_DIR
ENGINES = ["ostrich", "agx", "chrono"]
COLORS = {"ostrich": "#d62728", "agx": "#1f77b4", "chrono": "#2ca02c"}


def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


def final_table(md):
    """Headline roster re-scored at the current window (final_eval.py)."""
    d = load("final_eval.json")
    if not d:
        return
    md.append(f"## Headline: all configs at {d['window_s']:g} s windows\n")
    md.append("| config | GT-mean [m] | holdout (motors0) [m] |")
    md.append("|---|---|---|")
    for r in sorted(d["rows"], key=lambda r: r["holdout"]):
        md.append(f"| {r['label']} | {r['gt_mean']:.3f} | {r['holdout']:.3f} |")
    md.append("\n(Config selection came from the 15 s-window sweeps; this table "
              "re-scores those configs under the current window.)\n")


def sweep_table(md):
    md.append("## Axis A+B: sim-to-real accuracy (sweeps; scored with 15 s "
              "windows at sweep time)\n")
    md.append("| engine | defaults [m] | tuned best [m] | best params | "
              "holdout def [m] | holdout best [m] |")
    md.append("|---|---|---|---|---|---|")
    for e in ENGINES:
        s = load(f"sweep_{e}.json")
        if not s:
            continue
        h = s.get("holdout", {})
        hd = f"{h['defaults']['error']:.3f}" if h else "—"
        hb = f"{h['best']['error']:.3f}" if h else "—"
        md.append(f"| {e} | {s['defaults']['error']:.3f} | {s['best']['error']:.3f} "
                  f"| `{s['best']['params']}` | {hd} | {hb} |")
    md.append("")

    # Sensitivity: error spread across the stable grid rows per engine.
    md.append("### Solver sensitivity (spread across swept configs)\n")
    md.append("| engine | stable configs | unstable | min [m] | median [m] | max [m] |")
    md.append("|---|---|---|---|---|---|")
    for e in ENGINES:
        s = load(f"sweep_{e}.json")
        if not s:
            continue
        errs = [r["error"] for r in s["grid"] if r["stable"]]
        n_bad = sum(not r["stable"] for r in s["grid"])
        md.append(f"| {e} | {len(errs)} | {n_bad} | {min(errs):.3f} | "
                  f"{np.median(errs):.3f} | {max(errs):.3f} |")
    md.append("")


def extensions_table(md):
    """Wheel-terrain extensions: anisotropic friction, Chrono SCM, agxTerrain."""
    have = False
    lines = ["## Wheel-terrain extensions (real turn gain alpha ~ 2)\n",
             "| variant | GT-mean [m] | holdout [m] | alpha (1,3) | alpha mean | note |",
             "|---|---|---|---|---|---|"]
    for e in ("agx", "ostrich"):
        s = load(f"sweep_aniso_{e}.json")
        if not s or not s.get("best"):
            continue
        have = True
        b = s["best"]
        alphas = [a for a in b["alphas"] if a]
        hold = f"{b['holdout']:.3f}" if b.get("holdout") is not None else "—"
        lines.append(f"| {e} + anisotropic friction | {b['error']:.3f} | {hold} | "
                     f"{b['alphas'][0]:.2f} | {np.mean(alphas):.2f} | "
                     f"`{b['params']}` |")
    for name, label in (("scm_chrono_phi20.json", "chrono SCM (phi=20)"),
                        ("scm_chrono.json", "chrono SCM (phi=30)"),
                        ("agx_soil.json", "agxTerrain dirt_1 + TerrainWheel")):
        s = load(name)
        if not s:
            continue
        have = True
        alphas = [t["alpha"] for t in s["turn"] if t["alpha"]]
        hold = s["bags"].get(common.HOLDOUT_BAG, {}).get("combined_mean")
        note = "diverges mid-bag" if not all(
            b.get("stable", True) for b in s["bags"].values()) else ""
        lines.append(f"| {label} | {s['error_gt_mean']:.3f} | "
                     f"{hold:.3f} | {alphas[0]:.2f} | {np.mean(alphas):.2f} | {note} |")
    if have:
        md.extend(lines)
        md.append("")


def speed_plot(md):
    d = load("speed_dt.json")
    if not d:
        return
    rows = d["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for e in ENGINES:
        for cfg, ls in (("defaults", "--"), ("best", "-")):
            rr = sorted([r for r in rows if r["engine"] == e and r["config"] == cfg
                         and r["threads"] in (0, 1)], key=lambda r: r["dt"])
            if not rr:
                continue
            dts = [r["dt"] for r in rr]
            errs = [r["combined_mean"] if r["stable"] else np.nan for r in rr]
            rtfs = [r["rtf"] for r in rr]
            ax1.plot(dts, errs, ls, color=COLORS[e], marker="o", ms=4,
                     label=f"{e} {cfg}")
            ax2.plot(dts, rtfs, ls, color=COLORS[e], marker="o", ms=4)
    ax1.set_xscale("log"); ax1.set_xlabel("dt [s]")
    ax1.set_ylabel("window error [m]"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("dt [s]")
    ax2.set_ylabel("real-time factor"); ax2.grid(alpha=0.3)
    ax1.set_title(f"accuracy vs dt ({d['bag']})"); ax2.set_title("speed vs dt")
    fig.savefig(R / "speed_dt.png", dpi=110, bbox_inches="tight")

    md.append("## Axis C: speed / timestep (fast_experiment1)\n")
    md.append("![speed](speed_dt.png)\n")
    md.append("| engine | config | hardware | largest stable dt | RTF there | "
              "err there [m] |")
    md.append("|---|---|---|---|---|---|")
    for e in ENGINES:
        for cfg in ("defaults", "best"):
            rr = [r for r in rows if r["engine"] == e and r["config"] == cfg
                  and r["stable"] and r["threads"] in (0, 1)]
            if not rr:
                continue
            r = max(rr, key=lambda r: r["dt"])
            md.append(f"| {e} | {cfg} | {r['hardware']} | {r['dt']:g} | "
                      f"{r['rtf']:.1f}x | {r['combined_mean']:.3f} |")
    md.append("")


def scenario_table(md):
    d = load("scenarios.json")
    if not d:
        return
    rows = d["rows"]
    md.append("## Axis D: behavioral scenarios (best configs, 2 reps)\n")
    md.append("### step16 — 16 cm step climb (real robot: climbs)\n")
    md.append("| engine | config | cleared | t_clear [s] | max pitch [deg] |")
    md.append("|---|---|---|---|---|")
    for e in ENGINES:
        for cfg in ("defaults", "best"):
            rr = [r for r in rows if r["engine"] == e and r["config"] == cfg
                  and r["scenario"] == "step16"]
            for r in rr:
                tc = f"{r['time_to_clear_s']:.1f}" if r["time_to_clear_s"] else "—"
                md.append(f"| {e} | {cfg} | {r['cleared']} | {tc} | "
                          f"{r['max_pitch_deg']:.0f} |")
    md.append("\n### turn gain alpha (real robot on this surface: ~2)\n")
    md.append("| engine | config | " + " | ".join(f"({a},{b})" for a, b in
                                                  [(1, 3), (1.5, 3.5), (2, 4), (0.5, 3.5)]) + " |")
    md.append("|---|---|---|---|---|---|")
    for e in ENGINES:
        for cfg in ("defaults", "best"):
            rr = [r for r in rows if r["engine"] == e and r["config"] == cfg
                  and r["scenario"] == "turn_radius"]
            if not rr:
                continue
            by_pair = {}
            for r in rr:
                by_pair.setdefault(tuple(r["pair"]), []).append(r["alpha"])
            cells = []
            for pair in [(1.0, 3.0), (1.5, 3.5), (2.0, 4.0), (0.5, 3.5)]:
                a = by_pair.get(pair)
                cells.append(f"{np.mean([x for x in a if x]):.2f}" if a and any(a) else "—")
            md.append(f"| {e} | {cfg} | " + " | ".join(cells) + " |")
    md.append("\n### rock_field — 3x3 loose rocks\n")
    md.append("| engine | config | success | x_final [m] | lateral RMS [m] | wall [s] |")
    md.append("|---|---|---|---|---|---|")
    for e in ENGINES:
        for cfg in ("defaults", "best"):
            rr = [r for r in rows if r["engine"] == e and r["config"] == cfg
                  and r["scenario"] == "rock_field"]
            for r in rr:
                md.append(f"| {e} | {cfg} | {r['success']} | {r['x_final']:.1f} | "
                          f"{r['lateral_rms']:.2f} | {r['wall_clock_s']:.1f} |")
    md.append("")


def main():
    md = ["# Engine comparison — generated summary\n",
          f"(regenerate with plot_results.py; GT bags: {common.GT_BAGS}, "
          f"holdout: {common.HOLDOUT_BAG})\n"]
    final_table(md)
    sweep_table(md)
    extensions_table(md)
    speed_plot(md)
    scenario_table(md)
    out = R / "summary.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

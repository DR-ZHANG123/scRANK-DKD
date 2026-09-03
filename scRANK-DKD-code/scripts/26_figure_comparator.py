#!/usr/bin/env python
"""Figure of comparator construction and the reciprocal perturbation.

The experiments in 12b and 12c carried the paper's main claim but had no figure.
Reads only the result tables under results/tables and refits nothing.

Inputs: T41b_comparator_macro.tsv, T41c_comparator_pooled_delta.tsv,
        T42_cascade_necessity.tsv, T42b_cascade_pooled_delta.tsv
Output: figures/Fig14_comparator_construction.{pdf,png}"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scdrp import figstyle as FS                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"

COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}
CCOL = {"GLOM": FS.PALETTE[0], "TUB": FS.PALETTE[1]}

ARMS = [
    ("scPair_LASSO", "Single-cell-constrained\n(evaluated framework)"),
    ("matched", "Genome-wide,\nmatched cascade"),
    ("matched_direction_only", "Genome-wide, matched\ncascade, direction only"),
    ("unmatched", "Genome-wide,\n$\\chi^2$ top-$n$, no cascade"),
]


def fmt_p(p: float) -> str:
    return f"P = {p:.3f}" if p < 0.05 else f"P = {p:.2f}"


def panel_a(ax, macro: pd.DataFrame) -> None:
    """Macro AUROC per arm. Laid out horizontally; the arm names are long, so the left margin is wide."""
    FS.panel(ax, "a")
    yy = np.arange(len(ARMS))[::-1]
    for i, comp in enumerate(COMPS):
        row = macro[macro.compartment == comp].iloc[0]
        vals = [row[k] for k, _ in ARMS]
        ax.barh(yy + (0.5 - i) * 0.34, vals, height=0.3,
                color=CCOL[comp], alpha=0.85, edgecolor="#444444",
                linewidth=0.5, label=COMPS[comp], zorder=3)
        for y, v in zip(yy + (0.5 - i) * 0.34, vals):
            ax.text(v + 0.006, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=5.6, color="#333333")
    ax.set_yticks(yy)
    ax.set_yticklabels([lab for _, lab in ARMS], fontsize=5.9)
    ax.set_xlim(0.5, 1.03)
    ax.set_xlabel("Macro AUROC, leave-one-cohort-out", fontweight="bold",
                  fontsize=7)
    ax.axvline(0.5, color="#999999", lw=0.7, ls="--", zorder=1)
    ax.set_ylim(-0.62, len(ARMS) - 0.38)


def panel_b(ax, delta: pd.DataFrame) -> None:
    """Delta AUROC of scPair against each perturbed arm. The reversal of the conclusion lives in this panel."""
    FS.panel(ax, "b")
    keys = [k for k, _ in ARMS if k != "scPair_LASSO"]
    short = {"matched": "vs matched\ncascade",
             "matched_direction_only": "vs matched cascade,\ndirection only",
             "unmatched": "vs $\\chi^2$ top-$n$,\nno cascade"}
    yy = np.arange(len(keys))[::-1]
    for i, comp in enumerate(COMPS):
        sub = delta[delta.compartment == comp].set_index("comparator")
        d = sub.reindex(keys)
        y = yy + (0.5 - i) * 0.3
        ax.errorbar(d.delta, y, xerr=[d.delta - d.lo, d.hi - d.delta],
                    fmt="o", color=CCOL[comp], markersize=4.0, capsize=2.2,
                    lw=1.0, label=COMPS[comp], zorder=4)
        for yv, dv, pv in zip(y, d.delta, d.p_value):
            ax.text(d.hi.max() + 0.028, yv, fmt_p(pv), va="center",
                    ha="left", fontsize=5.6,
                    color="#C44E52" if pv < 0.05 else "#777777",
                    fontweight="bold" if pv < 0.05 else "normal")
    ax.axvline(0, color="#444444", lw=0.9, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([short[k] for k in keys], fontsize=5.9)
    ax.set_xlabel("$\\Delta$AUROC of single-cell constraint",
                  fontweight="bold", fontsize=7)
    ax.set_xlim(-0.10, 0.30)
    ax.set_ylim(-0.52, len(keys) - 0.40)


def panel_c(ax, casc: pd.DataFrame, cpool: pd.DataFrame) -> None:
    """Reciprocal test: the same cascade removed from the framework's own candidate pool."""
    FS.panel(ax, "c")
    wide = casc.pivot_table(index=["compartment", "held_out"], columns="arm",
                            values="auroc").reset_index()
    for i, comp in enumerate(COMPS):
        sub = wide[wide.compartment == comp]
        xs = np.array([0.0, 1.0]) + (i - 0.5) * 0.22
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.plot(xs, [r.sc_full_cascade, r.sc_no_cascade], "-o",
                    color=CCOL[comp], markersize=3.4, lw=0.9, alpha=0.8,
                    zorder=3, label=COMPS[comp] if j == 0 else None)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Full\ncascade", "Cascade\nremoved"], fontsize=6.4)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("AUROC in held-out cohort", fontweight="bold", fontsize=7)
    ax.set_ylim(0.72, 1.21)

    lines = []
    for comp in COMPS:
        r = cpool[cpool.compartment == comp].iloc[0]
        lines.append(f"{COMPS[comp][:4]}. $\\Delta$ = {r.delta:+.3f}, "
                     f"{fmt_p(r.p_value)}")
    ax.text(0.5, 0.995, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="top", fontsize=5.7, color="#333333",
            linespacing=1.6)


def main() -> None:
    FS.apply_style()
    macro = pd.read_csv(TAB / "T41b_comparator_macro.tsv", sep="\t")
    delta = pd.read_csv(TAB / "T41c_comparator_pooled_delta.tsv", sep="\t")
    casc = pd.read_csv(TAB / "T42_cascade_necessity.tsv", sep="\t")
    cpool = pd.read_csv(TAB / "T42b_cascade_pooled_delta.tsv", sep="\t")

    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.14, 1.04, 0.92], wspace=0.92,
                          bottom=0.20, top=0.92)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, macro)
    panel_b(fig.add_subplot(gs[0, 1]), delta)
    panel_c(fig.add_subplot(gs[0, 2]), casc, cpool)
    h, l = ax_a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7,
               handlelength=1.1, handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.012))
    FS.save(fig, FIG, "Fig14_comparator_construction")


if __name__ == "__main__":
    main()

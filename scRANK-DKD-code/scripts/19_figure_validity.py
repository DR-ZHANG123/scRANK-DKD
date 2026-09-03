#!/usr/bin/env python
"""Figure 8: where the cell-state labels hold, and where the cohorts differ.

Program attribution purity and control-tissue provenance, both read from the
confounding checks rather than recomputed."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import figstyle as FS                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
FIG = ROOT / "figures"

FS.apply_style()
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}


def figure8() -> None:
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(1, 2, wspace=0.46)

    ax = fig.add_subplot(gs[0, 0])
    FS.panel(ax, "a")
    spec = pd.read_csv(TAB / "T30_program_expression_specificity.tsv", sep="\t")
    used = spec[spec.used_in_candidate_pool].sort_values(
        "frac_genes_max_in_own_type")
    colors = ["#55A868" if f == "consistent" else "#C44E52"
              for f in used.attribution_flag]
    yy = np.arange(len(used))
    ax.barh(yy, used.frac_genes_max_in_own_type, color=colors,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(yy)
    ax.set_yticklabels([p.replace("209781_", "").replace("131882_", "")
                        .replace("_DE", "") for p in used.program], fontsize=4.4)
    ax.set_xlabel("Program genes with highest mean expression\n"
                  "in the assigned cell type", fontweight="bold")
    ax.set_xlim(0, max(0.6, float(used.frac_genes_max_in_own_type.max()) * 1.15))
    n_re = int((used.attribution_flag == "reassigned").sum())
    ax.text(0.97, 0.03, f"{n_re}/{len(used)} programs reassigned",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            color="#C44E52")

    ax = fig.add_subplot(gs[0, 1])
    FS.panel(ax, "b")
    prov = pd.read_csv(TAB / "T28_control_provenance.tsv", sep="\t")
    prov["label"] = (prov.cohort.map(FS.cohort) + "\n(" +
                     prov.compartment.str.lower() + ")")
    xx = np.arange(len(prov))
    ax.bar(xx, prov["tumour nephrectomy"], width=0.62, color="#DD8452",
           edgecolor="white", linewidth=0.3, label="Tumour nephrectomy")
    ax.bar(xx, prov["living donor"], bottom=prov["tumour nephrectomy"],
           width=0.62, color="#4C72B0", edgecolor="white", linewidth=0.3,
           label="Living donor")
    rest = (prov.n_control - prov["tumour nephrectomy"] -
            prov["living donor"]).clip(lower=0)
    ax.bar(xx, rest, bottom=prov["tumour nephrectomy"] + prov["living donor"],
           width=0.62, color="#BBBBBB", edgecolor="white", linewidth=0.3,
           label="Not specified")
    ax.set_xticks(xx)
    ax.set_xticklabels(prov.label, fontsize=5.4, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Control samples", fontweight="bold")
    ax.set_ylim(0, prov.n_control.max() * 1.5)
    FS.legend(ax, loc="upper left", handlelength=1.0, fontsize=5.6)

    FS.save(fig, FIG, "Fig8_validity_checks")


if __name__ == "__main__":
    figure8()

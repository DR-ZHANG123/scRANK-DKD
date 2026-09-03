#!/usr/bin/env python
"""Interpretation (1): collapse the pair model into a per-gene directed shift.

A 79,800-dimensional pair space is unreadable. This reduces it to something a
person can inspect: for each retained pair (A, B) the direction is
P = 1[rank(A) > rank(B)] and delta = reversal_DKD - reversal_Control. A positive
delta means A more often outranks B in DKD, so A rises relative to B and B falls
relative to A. Summing per gene:

    net_shift(g) = sum over pairs with g=A of w*delta
                 - sum over pairs with g=B of w*delta

where w is the pair's bootstrap stability. A positive net_shift means the gene
rises in DKD.

The point is that this ordering uses no cell attribution at all, so it is
unaffected by the doubtful cell-of-origin labels this study documents, and its
reproducibility across folds can be tested directly.

Writes T31 (per-gene shift and cross-fold agreement) and Fig9."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import figstyle as FS                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"

FS.apply_style()
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}


def net_shift(pairs: pd.DataFrame) -> pd.Series:
    """Signed, stability-weighted sum of pair directions, per gene."""
    w = pairs["stability"] if "stability" in pairs else pairs["mean_stability"]
    contrib_a = pairs.assign(g=pairs.gene_a, s=w * pairs.get("delta",
                             pairs.get("mean_delta")))
    contrib_b = pairs.assign(g=pairs.gene_b, s=-w * pairs.get("delta",
                             pairs.get("mean_delta")))
    both = pd.concat([contrib_a[["g", "s"]], contrib_b[["g", "s"]]])
    return both.groupby("g").s.sum().sort_values()


def per_fold_shift(comp: str) -> pd.DataFrame:
    """Per-gene net shift computed separately within each outer fold."""
    t12 = pd.read_csv(TAB / "T12_selected_pairs.tsv.gz", sep="\t")
    t12 = t12[t12.compartment == comp]
    out = {}
    for held, sub in t12.groupby("held_out"):
        out[held] = net_shift(sub)
    return pd.DataFrame(out)


def fold_reproducibility(fold_shifts: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation and sign agreement of the per-gene shift between folds."""
    cols = list(fold_shifts.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = fold_shifts[cols[i]], fold_shifts[cols[j]]
            common = a.dropna().index.intersection(b.dropna().index)
            if len(common) < 10:
                continue
            rho = float(spearmanr(a[common], b[common]).statistic)
            rows.append(dict(fold_a=cols[i], fold_b=cols[j],
                             n_shared_genes=len(common), spearman=round(rho, 3)))
    return pd.DataFrame(rows)


def main() -> None:
    all_shift, all_repro = [], []
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    fig.subplots_adjust(hspace=0.62, wspace=0.42, top=0.95,
                        bottom=0.09, left=0.13, right=0.97)

    for row, comp in enumerate(COMPS):
        core = pd.read_csv(TAB / "T16_core_pairs.tsv.gz", sep="\t")
        core = core[(core.compartment == comp) & (core.fold_consistency >= 2 / 3)]
        shift = net_shift(core).rename("net_shift").to_frame()
        shift["compartment"] = comp

        fs = per_fold_shift(comp)
        repro = fold_reproducibility(fs)
        repro["compartment"] = comp
        sign = np.sign(fs)
        maj = sign.apply(lambda r: np.nan if r.dropna().empty
                         else r.dropna().mode().iloc[0], axis=1)
        agree = sign.eq(maj, axis=0).sum(axis=1) / sign.notna().sum(axis=1)
        shift["fold_sign_agreement"] = agree.reindex(shift.index).values
        shift["n_folds_present"] = sign.notna().sum(axis=1).reindex(
            shift.index).values
        all_shift.append(shift.reset_index().rename(columns={"index": "gene",
                                                             "g": "gene"}))
        all_repro.append(repro)

        ax = axes[row, 0]
        FS.panel(ax, "ab"[row])
        s = shift.net_shift.dropna().sort_values()
        pick = pd.concat([s.head(10), s.tail(10)])
        colors = ["#4C72B0" if v < 0 else "#C44E52" for v in pick.values]
        ax.barh(range(len(pick)), pick.values, color=colors,
                edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(pick)))
        ax.set_yticklabels(pick.index, fontsize=5.4)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_xlabel("Net rank shift in DKD (down / up)", fontweight="bold")
        ax.set_title(COMPS[comp], fontsize=7.5, pad=4, loc="center")

        ax = axes[row, 1]
        FS.panel(ax, "cd"[row])
        vals = shift.fold_sign_agreement.dropna()
        ax.hist(vals, bins=np.linspace(0.4, 1.01, 13), color=FS.PALETTE[row],
                edgecolor="white", linewidth=0.3)
        ax.axvline(float(vals.median()), color="#444444", ls="--", lw=0.9)
        ax.text(0.03, 0.96, f"median {vals.median():.2f}\n"
                f"n = {len(vals)} genes", transform=ax.transAxes, va="top",
                fontsize=6.2)
        ax.set_xlabel("Fraction of folds agreeing\non shift direction", fontweight="bold")
        ax.set_ylabel("Number of genes", fontweight="bold")

    FS.save(fig, FIG, "Fig9_directed_rank_graph")

    shift_tab = pd.concat(all_shift, ignore_index=True)
    shift_tab.to_csv(TAB / "T31_gene_net_shift.tsv", sep="\t", index=False)
    repro_tab = pd.concat(all_repro, ignore_index=True)
    repro_tab.to_csv(TAB / "T32_shift_fold_reproducibility.tsv", sep="\t",
                     index=False)

    print("===== cross-fold reproducibility (Spearman of the net directed shift) =====")
    print(repro_tab.to_string(index=False))
    print("\n===== per-compartment cross-fold sign agreement =====")
    for comp, sub in shift_tab.groupby("compartment"):
        v = sub.fold_sign_agreement.dropna()
        print(f"  {comp}: median {v.median():.2f}, "
              f"≥ 2/3 folds agree: {(v >= 2/3 - 1e-9).mean():.2f} of "
              f"{len(v)} genes")
    print("\n===== top rising / falling genes per compartment =====")
    for comp, sub in shift_tab.groupby("compartment"):
        s = sub.set_index("gene").net_shift.dropna().sort_values()
        print(f"  {comp} falling in DKD: {list(s.head(6).index)}")
        print(f"  {comp} rising in DKD: {list(s.tail(6).index[::-1])}")


if __name__ == "__main__":
    main()

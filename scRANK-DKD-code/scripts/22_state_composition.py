#!/usr/bin/env python
"""[EXPLORATORY - supports no claim in the manuscript] Interpretation (3): split the
model signal into cell-state change and cell-composition change.

Earlier there was only a qualitative statement (adjusting for cell proportions
removed the tubulointerstitial score's significance). This gives the quantitative
decomposition. For each candidate gene the bulk-level change between DKD and
control can be written (Bhattacherjee-type decomposition) as:

  delta_g = sum_c pbar_c*dmu_gc  +  sum_c dp_c*mubar_gc  +  sum_c dp_c*dmu_gc
            |___ state term ___|     |_ composition _|      |_ interaction _|

where mu_gc is the mean expression of gene g in cell type c, p_c is the proportion
of cell type c, delta is DKD minus control, and a bar denotes the mean of the two
groups. All three terms are estimated directly from the single-cell data, without
going through the unreliable bulk deconvolution.

The terms are then weighted by how strongly the model uses each gene (the absolute
directed shift |net_shift| from T31), giving the share of the model signal that
comes from composition change.

Single-cell proportions are affected by dissociation bias and do not match the
composition of microdissected bulk tissue, so this decomposition is indicative
only; the manuscript says so explicitly.

Writes T35 (per-gene decomposition), T36 (model-weighted summary) and Fig11."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import figstyle as FS                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data_processed" / "scrna"
PAIR = ROOT / "data_processed" / "pair_matrix"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"

FS.apply_style()
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}
DROP_CT = {"LowQuality", "Unassigned"}


def cell_profiles(ds: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Mean expression per cell type and the cell-type proportions, per disease group.
    
    Estimated directly from the single-cell data. Both are needed for all three terms
    of the decomposition."""
    ad = sc.read_h5ad(SCP / f"{ds}_annotated.h5ad")
    expr = ad.raw.to_adata() if ad.raw is not None else ad
    keep = ~ad.obs.cell_type.isin(DROP_CT).values
    cts = sorted(pd.unique(ad.obs.cell_type[keep]))

    mu = {"Control": {}, "DKD": {}}
    frac = {"Control": {}, "DKD": {}}
    for grp in ("Control", "DKD"):
        gmask = keep & (ad.obs.group.values == grp)
        n_tot = int(gmask.sum())
        for ct in cts:
            m = gmask & (ad.obs.cell_type.values == ct)
            frac[grp][ct] = m.sum() / max(1, n_tot)
            X = expr[m].X
            mu[grp][ct] = (np.asarray(X.mean(axis=0)).ravel() if m.sum()
                           else np.zeros(expr.n_vars))
    mu_c = pd.DataFrame(mu["Control"], index=expr.var_names)
    mu_d = pd.DataFrame(mu["DKD"], index=expr.var_names)
    dp = pd.Series(frac["DKD"], dtype=float) - pd.Series(frac["Control"],
                                                         dtype=float)
    return mu_c, mu_d, dp


def decompose(genes: list[str], ds: str) -> pd.DataFrame:
    mu_c, mu_d, dp = cell_profiles(ds)
    p_bar = None
    ad = sc.read_h5ad(SCP / f"{ds}_annotated.h5ad")
    keep = ~ad.obs.cell_type.isin(DROP_CT).values
    cts = list(mu_c.columns)
    pc = {}
    for ct in cts:
        pc[ct] = ((keep & (ad.obs.cell_type.values == ct)).sum()
                  / max(1, keep.sum()))
    p_bar = pd.Series(pc)

    g = [x for x in genes if x in mu_c.index]
    mc, md = mu_c.loc[g], mu_d.loc[g]
    dmu = md - mc
    mu_bar = (mc + md) / 2

    state = (dmu * p_bar).sum(axis=1)
    comp = (mu_bar * dp).sum(axis=1)
    inter = (dmu * dp).sum(axis=1)
    total = state + comp + inter
    return pd.DataFrame(dict(gene=g, state=state.values, composition=comp.values,
                             interaction=inter.values, total=total.values))


def main() -> None:
    all_gene, summary = [], []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    net = pd.read_csv(TAB / "T31_gene_net_shift.tsv", sep="\t")

    for ci, comp_name in enumerate(COMPS):
        cand = pd.read_csv(PAIR / f"{comp_name}_candidate_genes.tsv", sep="\t")
        prog = pd.read_csv(ROOT / "data_processed" / "programs" /
                           "programs_raw.tsv", sep="\t")
        src = prog.set_index("program")["source_dataset"].to_dict() \
            if "source_dataset" in prog.columns else {}
        cand["ds"] = cand.program.map(lambda p: src.get(p, "GSE209781"))

        parts = []
        for ds, sub in cand.groupby("ds"):
            dec = decompose(list(sub.gene), ds)
            parts.append(dec)
        dec = pd.concat(parts, ignore_index=True).drop_duplicates("gene")
        dec["compartment"] = comp_name
        denom = dec.state.abs() + dec.composition.abs()
        dec["state_fraction"] = np.where(denom > 0, dec.state.abs() / denom,
                                         np.nan)
        all_gene.append(dec)

        w = (net[net.compartment == comp_name]
             .set_index("gene").net_shift.abs())
        d = dec.set_index("gene")
        common = d.index.intersection(w.index)
        wts = w.reindex(common).fillna(0.0)
        wsum = wts.sum()
        state_w = (d.loc[common].state.abs() * wts).sum() / max(wsum, 1e-9)
        comp_w = (d.loc[common].composition.abs() * wts).sum() / max(wsum, 1e-9)
        inter_w = (d.loc[common].interaction.abs() * wts).sum() / max(wsum, 1e-9)
        tot = state_w + comp_w + inter_w
        summary.append(dict(compartment=comp_name, n_genes_used=len(common),
                            state_pct=round(100 * state_w / tot, 1),
                            composition_pct=round(100 * comp_w / tot, 1),
                            interaction_pct=round(100 * inter_w / tot, 1)))

        ax = axes[ci]
        FS.panel(ax, "ab"[ci])
        top = w.reindex(common).sort_values(ascending=False).head(18).index[::-1]
        dd = d.loc[top]
        yy = np.arange(len(top))
        ax.barh(yy, dd.state, color="#C44E52", edgecolor="white",
                linewidth=0.3, label="Cell-state")
        ax.barh(yy, dd.composition, left=dd.state, color="#4C72B0",
                edgecolor="white", linewidth=0.3, label="Composition")
        ax.barh(yy, dd.interaction, left=dd.state + dd.composition,
                color="#8C8C8C", edgecolor="white", linewidth=0.3,
                label="Interaction")
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels(top, fontsize=5.0)
        ax.set_xlabel("Predicted bulk change, decomposed", fontweight="bold")
        sm = summary[-1]
        ax.set_title(f"{COMPS[comp_name]}: state {sm['state_pct']:.0f}% / "
                     f"comp {sm['composition_pct']:.0f}%", fontsize=6.6, pad=3)
        if ci == 0:
            FS.legend(ax, loc="lower right", fontsize=5.6, handlelength=1.0)

    FS.save(fig, FIG, "Fig11_state_composition")

    gene_tab = pd.concat(all_gene, ignore_index=True)
    gene_tab.to_csv(TAB / "T35_state_composition_gene.tsv", sep="\t",
                    index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(TAB / "T36_state_composition_summary.tsv", sep="\t",
                index=False)
    print("===== model-weighted share of state / composition / interaction =====")
    print(summ.to_string(index=False))
    print("\n===== median per-gene state share per compartment =====")
    for comp, sub in gene_tab.groupby("compartment"):
        v = sub.state_fraction.dropna()
        print(f"  {comp}: median state fraction {v.median():.2f} "
              f"(n={len(v)} genes)")


if __name__ == "__main__":
    main()

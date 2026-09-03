#!/usr/bin/env python
"""Interpretation (2): data-driven ablation of functional modules.

The question is whether the classification signal concentrates on one functional
axis (extracellular matrix and fibrosis, say) or is spread out. It does not depend
on whether the cell attribution is correct, so it is unaffected by the doubtful
cell-of-origin labels this study documents.

Procedure:
1. Correlate the candidate genes by within-sample percentile rank across all LOCO
   training samples and cut a hierarchical clustering into co-expression modules;
   module definition touches no DKD label.
2. Name each module automatically by its overlap with canonical marker sets.
3. Leave-one-module-out: drop all of a module's genes, rerun screening and
   scPair-LASSO, and record the drop in held-out AUROC.
4. Single-module retention: use only that module's genes and see how far one axis gets.

Writes T33 (module definitions), T34 (ablation results) and Fig10."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                          # noqa: E402
from scdrp import data as D                                # noqa: E402
from scdrp import figstyle as FS                           # noqa: E402
from scdrp import screening as SC                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"

FS.apply_style()
SEED = 20260722
N_MODULES = 8
N_PAIRS = 100
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}

MARKERS = {
    "ECM/fibrosis": ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1",
                     "COL6A1", "COL15A1", "FN1", "LUM", "DCN", "POSTN",
                     "TIMP1", "MMP2", "MMP7", "SPARC", "BGN", "VCAN",
                     "TGFBI", "SERPINH1", "PAPPA2", "THBS1", "THBS2"],
    "immune": ["CD68", "CD3D", "LYZ", "C1QA", "C1QB", "CD74",
                            "HLA-DRA", "ITGAM", "CSF1R", "CCL2", "CXCL6",
                            "IL32", "TYROBP", "AIF1", "LTF", "NKG7"],
    "complement": ["C7", "C1S", "C1R", "C3", "CFH", "CFB", "SERPING1"],
    "IEG/stress": ["FOS", "JUN", "JUNB", "EGR1", "ATF3", "NR4A1",
                               "NR4A2", "KLF2", "KLF6", "KLF10", "DUSP1",
                               "BTG2", "ZFP36", "HSPA1A", "HSPA1L", "HSPH1",
                               "DNAJB1", "DNAJB4", "RHOB", "GADD45B"],
    "tubular metab.": ["LRP2", "CUBN", "SLC34A1", "SLC12A1",
                                     "SLC12A3", "UMOD", "MIOX", "ALDOB",
                                     "PCK1", "GATM", "GGT5", "ACACB", "PDK4",
                                     "AQP1", "AQP2", "RHCG", "ATP1A1",
                                     "GPX3", "ADH1B", "EGF"],
}


def build_modules(comp: D.Compartment) -> pd.Series:
    """Cluster candidate genes into co-expression modules from within-sample rank correlation."""
    cds = list(comp.cohorts.values())
    genes = list(comp.genes)
    mat = []
    for c in cds:
        r = c.rank.reindex(genes)
        mat.append(r.values)
    R = np.hstack(mat)
    ok = np.isfinite(R).mean(axis=1) > 0.8
    R = np.where(np.isfinite(R), R, np.nan)
    corr = pd.DataFrame(R[ok].T, columns=np.array(genes)[ok]).corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(Z, t=N_MODULES, criterion="maxclust")
    return pd.Series(labels, index=np.array(genes)[ok], name="module")


def name_module(members: list[str]) -> tuple[str, float]:
    """Name a module by its overlap with canonical marker sets.
    
    A module is named after the marker set it overlaps most, provided the overlap is
    large enough to be meaningful; otherwise it keeps a numeric label. Naming is
    cosmetic and enters no computation."""
    mset = set(members)
    best, best_frac, best_hits = "no marker enrichment", 0.0, 0
    for label, mk in MARKERS.items():
        hits = len(mset & set(mk))
        frac = hits / max(1, len(mk))
        if hits >= 3 and frac > best_frac:
            best, best_frac, best_hits = label, frac, hits
    tag = best if best_hits >= 3 else "no marker enrichment"
    return tag, best_frac


def ablate(comp: D.Compartment, drop_genes: set[str] | None,
           keep_genes: set[str] | None) -> float:
    """Rerun screening and the classifier with one module's genes removed, or with only that module."""
    genes = list(comp.genes)
    gi = {g: i for i, g in enumerate(genes)}
    ga = comp.pairs.gene_a.values
    gb = comp.pairs.gene_b.values
    if keep_genes is not None:
        keep_pair = np.array([a in keep_genes and b in keep_genes
                              for a, b in zip(ga, gb)])
    else:
        keep_pair = np.array([a not in drop_genes and b not in drop_genes
                              for a, b in zip(ga, gb)])
    pair_idx = np.flatnonzero(keep_pair)
    if len(pair_idx) < 50:
        return float("nan")
    sub_pairs = comp.pairs.iloc[pair_idx].reset_index(drop=True)
    sub_pairs["pair_id"] = np.arange(len(sub_pairs))

    aucs = []
    for held in comp.cohorts:
        tr = [c for k, c in comp.cohorts.items() if k != held]
        te = comp.cohorts[held]
        P, M, mask, y, cid = D.stack(tr, pair_idx)
        sel = SC.select_pairs(P, M, mask, y, cid, sub_pairs, n_max=N_PAIRS,
                              seed=SEED)
        if sel.empty:
            aucs.append(0.5)
            continue
        keep = sel.pair_id.values
        Xtr = np.hstack([np.where(mask[keep], P[keep], 0.5).T,
                         np.where(mask[keep], M[keep], 0.0).T])
        Pte, Mte, mte, yte, _ = D.stack([te], pair_idx)
        Xte = np.hstack([np.where(mte[keep], Pte[keep], 0.5).T,
                         np.where(mte[keep], Mte[keep], 0.0).T])
        m = bl.fit_lasso_cv(Xtr, y)
        aucs.append(float(roc_auc_score(te.y, m.predict_proba(Xte)[:, 1])))
    return float(np.mean(aucs))


def main() -> None:
    mod_rows, abl_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    fig.subplots_adjust(wspace=0.75, left=0.19, right=0.99,
                        bottom=0.16, top=0.90)

    for ci, comp_name in enumerate(COMPS):
        comp = D.load_compartment(comp_name)
        full = ablate(comp, set(), None)
        modules = build_modules(comp)

        named = {}
        for m in sorted(modules.unique()):
            members = list(modules[modules == m].index)
            label, score = name_module(members)
            key = f"CM{m} ({label})" if label != "no marker enrichment" \
                else f"CM{m}"
            named[m] = key
            mod_rows.append(dict(compartment=comp_name, module=m, label=label,
                                 n_genes=len(members),
                                 top_genes=",".join(members[:12])))

        results = []
        for m in sorted(modules.unique()):
            g = set(modules[modules == m].index)
            drop_auc = ablate(comp, g, None)
            keep_auc = ablate(comp, None, g)
            results.append(dict(compartment=comp_name, module=named[m],
                                n_genes=len(g),
                                full_auroc=round(full, 3),
                                leave_out_auroc=round(drop_auc, 3),
                                drop=round(full - drop_auc, 3),
                                keep_only_auroc=round(keep_auc, 3)))
        abl = pd.DataFrame(results).sort_values("drop", ascending=False)
        abl_rows.append(abl)

        ax = axes[ci]
        FS.panel(ax, "ab"[ci])
        yy = np.arange(len(abl))
        ax.barh(yy - 0.2, abl["drop"], height=0.36, color="#C44E52",
                edgecolor="white", linewidth=0.3, label="AUROC lost if removed")
        ax.barh(yy + 0.2, abl["keep_only_auroc"] - 0.5, height=0.36,
                color="#4C72B0", edgecolor="white", linewidth=0.3,
                label="AUROC $-$ 0.5 if kept alone")
        ax.axvline(0, color="#444444", lw=0.8)
        ax.axvline(full - 0.5, color="#4C72B0", ls=":", lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{m}  (n={n})"
                            for m, n in zip(abl.module, abl.n_genes)],
                           fontsize=5.6)
        ax.set_xlabel("Change in macro AUROC", fontweight="bold")
        ax.set_title(f"{COMPS[comp_name]} (all modules, fixed 100 pairs: "
                     f"{full:.3f})", fontsize=7, pad=3)
        if ci == 0:
            FS.legend(ax, loc="lower right", fontsize=5.6, handlelength=1.0)

    FS.save(fig, FIG, "Fig10_module_ablation")

    pd.concat(mod_rows and [pd.DataFrame(mod_rows)], ignore_index=True).to_csv(
        TAB / "T33_coexpression_modules.tsv", sep="\t", index=False)
    abl_all = pd.concat(abl_rows, ignore_index=True)
    abl_all.to_csv(TAB / "T34_module_ablation.tsv", sep="\t", index=False)
    print("===== leave-one-module-out ablation =====")
    print(abl_all.to_string(index=False))


if __name__ == "__main__":
    main()

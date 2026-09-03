#!/usr/bin/env python
"""Figures 1-3: study design, single-cell atlas, cell-state-constrained rank pairs."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import figstyle as FS                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
BULK = ROOT / "data_processed" / "bulk"
SCP = ROOT / "data_processed" / "scrna"
PAIRD = ROOT / "data_processed" / "pair_matrix"
FIG = ROOT / "figures"

FS.apply_style()


# --------------------------------------------------------------------------- #
def figure1() -> None:
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], hspace=0.30,
                          wspace=0.34)

    ax = fig.add_subplot(gs[0, :])
    FS.panel(ax, "a")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.16, 1.02)
    rows = [
        [("Human DKD single-cell\nand single-nucleus RNA-seq\n"
          "GSE131882 + GSE209781\n(12 donors, 59k cells)", "#4C72B0"),
         ("Donor-level pseudo-bulk\ndifferential expression\n"
          "+ consensus NMF\nper cell type", "#55A868"),
         ("Cell-state programs\nwith explicit cell of origin\n"
          "and direction\n(30 programs, 400 genes)", "#8172B3")],
        [("Within-sample percentile\nrank in each bulk cohort\n"
          "no cross-cohort\nbatch correction", "#DD8452"),
         ("Cell-state-constrained\ngene rank pairs\n"
          "screened for reversal,\nconsistency, stability", "#C44E52"),
         ("Patient-level models\nscPair-LASSO and\nprogram-aware DeepSets\n"
          "leave-one-cohort-out", "#64B5CD")],
    ]
    bw, bh = 0.285, 0.30
    for r, row in enumerate(rows):
        ytop = 0.66 - r * 0.40
        for i, (text, color) in enumerate(row):
            x = 0.035 + i * (bw + 0.045)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, ytop), bw, bh, boxstyle="round,pad=0.008",
                facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.0))
            ax.text(x + bw / 2, ytop + bh / 2, text, ha="center", va="center",
                    fontsize=5.9, color="#222222", linespacing=1.35)
            if i < len(row) - 1:
                ax.annotate("", xy=(x + bw + 0.040, ytop + bh / 2),
                            xytext=(x + bw + 0.005, ytop + bh / 2),
                            arrowprops=dict(arrowstyle="-|>", color="#666666",
                                            linewidth=1.0))
        if r == 0:
            y_mid = ytop - 0.05
            ax.annotate("", xy=(0.035 + bw / 2, ytop - 0.10),
                        xytext=(0.035 + 2 * (bw + 0.045) + bw / 2, y_mid),
                        arrowprops=dict(arrowstyle="-|>", color="#666666",
                                        linewidth=1.0,
                                        connectionstyle="angle,angleA=-90,"
                                                        "angleB=180,rad=6"))

    ax = fig.add_subplot(gs[1, 0])
    FS.panel(ax, "b")
    coh = pd.read_csv(TAB / "T02_cohort_definition.tsv", sep="\t")
    coh = coh[coh.in_loco].copy()
    coh["label"] = coh.cohort.map(FS.cohort)
    ypos = np.arange(len(coh))
    ax.barh(ypos, coh.n_DKD, height=0.66, color="#C44E52", edgecolor="white",
            linewidth=0.3, label="DKD")
    ax.barh(ypos, coh.n_Control, left=coh.n_DKD, height=0.66, color="#4C72B0",
            edgecolor="white", linewidth=0.3, label="Control")
    for i, r in enumerate(coh.itertuples()):
        ax.text(r.n_DKD + r.n_Control + 1.5, i, f"{r.n_DKD}/{r.n_Control}",
                va="center", fontsize=6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.label} ({r.compartment[:4]})"
                        for r in coh.itertuples()], fontsize=6)
    ax.set_xlabel("Biopsy samples (DKD / control)", fontweight="bold")
    ax.set_xlim(0, coh[["n_DKD", "n_Control"]].sum(axis=1).max() * 1.34)
    FS.legend(ax, loc="center right")

    ax = fig.add_subplot(gs[1, 1])
    FS.panel(ax, "c")
    plat = coh
    colors = [FS.PALETTE[0] if c == "glomerulus" else FS.PALETTE[1]
              for c in plat.compartment]
    ax.scatter(plat.n_genes / 1000, plat.n_total, c=colors, s=44,
               edgecolor="white", linewidth=0.6, zorder=3)
    offs = {"GPL571": (6, -3), "GPL17586": (-46, 4), "GPL19184": (-49, -12),
            "GPL24120": (8, 2), "GPL22945": (7, -9)}
    seen = set()
    for r in plat.itertuples():
        if r.gpl in seen:
            continue
        seen.add(r.gpl)
        ax.annotate(r.gpl, (r.n_genes / 1000, r.n_total),
                    textcoords="offset points",
                    xytext=offs.get(r.gpl, (7, 7)), fontsize=5.8,
                    color="#444444")
    ax.set_xlabel("Genes on platform (thousands)", fontweight="bold")
    ax.set_ylabel("Samples in cohort", fontweight="bold")
    ax.set_xlim(8.5, 25.5)
    ax.set_ylim(0, plat.n_total.max() * 1.35)
    handles = [mpatches.Patch(color=FS.PALETTE[0], label="Glomerulus"),
               mpatches.Patch(color=FS.PALETTE[1], label="Tubulointerstitium")]
    FS.legend(ax, handles=handles, loc="upper right")

    FS.save(fig, FIG, "Fig1_study_design")


# --------------------------------------------------------------------------- #
def _place_labels(ax, coords: dict[str, tuple[float, float]],
                  fontsize: float = 4.8, n_iter: int = 220) -> None:
    """Place cluster labels at population centroids, nudging them apart to avoid overlap."""
    keys = list(coords)
    pos = np.array([coords[k] for k in keys], dtype=float)
    span = pos.max(0) - pos.min(0)
    min_d = 0.155 * np.linalg.norm(span) / max(1, np.sqrt(len(keys)))
    for _ in range(n_iter):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    push = (d / (dist + 1e-9)) * (min_d - dist) * 0.5
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    for k, (x, y) in zip(keys, pos):
        ax.text(x, y, k, fontsize=fontsize, ha="center", va="center",
                color="#111111", zorder=5,
                bbox=dict(facecolor="white", alpha=0.72, pad=0.6,
                          edgecolor="none"))


def figure2() -> None:
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.52)

    for row, ds in enumerate(("GSE131882", "GSE209781")):
        ad = sc.read_h5ad(SCP / f"{ds}_annotated.h5ad")
        um = ad.obsm["X_umap"]
        keep = ~ad.obs.cell_type.isin(["LowQuality", "Unassigned"]).values

        ax = fig.add_subplot(gs[row, 0])
        FS.panel(ax, "a" if row == 0 else "b")
        types = sorted(pd.unique(ad.obs.cell_type[keep]))
        cmap = {t: FS.PALETTE[i % len(FS.PALETTE)] for i, t in enumerate(types)}
        label_pos = {}
        for t in types:
            m = keep & (ad.obs.cell_type.values == t)
            ax.scatter(um[m, 0], um[m, 1], s=0.4, alpha=0.55,
                       color=cmap[t], linewidths=0, rasterized=True)
            if m.sum() / keep.sum() >= 0.015:
                label_pos[t] = (np.median(um[m, 0]), np.median(um[m, 1]))
        _place_labels(ax, label_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontweight="bold")
        ax.set_ylabel("UMAP 2", fontweight="bold")
        ax.text(0.02, 0.98,
                f"{ds}\n{int(keep.sum()):,} of {ad.n_obs:,} cells shown",
                fontsize=6, transform=ax.transAxes, va="top")

        ax = fig.add_subplot(gs[row, 1])
        FS.panel(ax, "c" if row == 0 else "d")
        for g, col in (("Control", "#4C72B0"), ("DKD", "#C44E52")):
            m = keep & (ad.obs.group.values == g)
            ax.scatter(um[m, 0], um[m, 1], s=0.4, alpha=0.45, color=col,
                       linewidths=0, rasterized=True, label=g)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontweight="bold")
        ax.set_ylabel("UMAP 2", fontweight="bold")
        FS.legend(ax, loc="upper right", markerscale=6, handletextpad=0.4)

        ax = fig.add_subplot(gs[row, 2])
        FS.panel(ax, "e" if row == 0 else "f")
        comp = pd.read_csv(TAB / f"T05_celltype_composition_{ds}.tsv",
                           sep="\t", index_col=0)
        comp = comp.drop(columns=[c for c in ("LowQuality", "Unassigned")
                                  if c in comp.columns])
        frac = comp.div(comp.sum(axis=1), axis=0).T
        frac = frac.loc[frac.mean(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(frac.values, aspect="auto", cmap="Blues",
                       vmin=0, vmax=float(frac.values.max()))
        ax.set_xticks(range(frac.shape[1]))
        ax.set_xticklabels(frac.columns, rotation=90, fontsize=6.2)
        ax.set_yticks(range(frac.shape[0]))
        ax.set_yticklabels(frac.index, fontsize=6.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("Fraction of cells", fontsize=6.6)

    ax = fig.add_subplot(gs[2, 0])
    FS.panel(ax, "g")
    prog = pd.read_csv(TAB / "T08_programs.tsv", sep="\t")
    de1 = pd.read_csv(TAB / "T06_pseudobulk_de_GSE131882.tsv.gz", sep="\t")
    de2 = pd.read_csv(TAB / "T06_pseudobulk_de_GSE209781.tsv.gz", sep="\t")
    from scipy.stats import spearmanr
    rows = []
    for ct in sorted(set(de1.cell_type) & set(de2.cell_type)):
        if ct in ("LowQuality", "Unassigned"):
            continue
        a = de1[(de1.cell_type == ct) & (de1.pvalue < 0.01)]
        b = de2[de2.cell_type == ct]
        m = a[["gene", "log2FoldChange"]].merge(
            b[["gene", "log2FoldChange"]], on="gene",
            suffixes=("_1", "_2")).dropna()
        if len(m) < 20:
            continue
        rho = float(spearmanr(m.log2FoldChange_1, m.log2FoldChange_2).statistic)
        rows.append(dict(cell_type=ct, rho=rho, n=len(m)))
    rep = pd.DataFrame(rows).sort_values("rho")
    ax.barh(range(len(rep)), rep.rho,
            color=["#C44E52" if v < 0 else "#55A868" for v in rep.rho],
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#444444", lw=0.8)
    for i, r in enumerate(rep.itertuples()):
        off = 0.02 if r.rho >= 0 else -0.02
        ax.text(r.rho + off, i, f"n={r.n}", va="center",
                ha="left" if r.rho >= 0 else "right", fontsize=6.2)
    ax.set_yticks(range(len(rep)))
    ax.set_yticklabels(rep.cell_type, fontsize=6.6)
    ax.set_xlabel("Cross-dataset log$_2$FC ρ", fontweight="bold")
    ax.set_xlim(min(-0.25, rep.rho.min() * 1.5), max(0.55, rep.rho.max() * 1.5))

    ax = fig.add_subplot(gs[2, 1])
    FS.panel(ax, "h")
    frames = []
    for ds in ("GSE131882", "GSE209781"):
        de = pd.read_csv(TAB / f"T06_pseudobulk_de_{ds}.tsv.gz", sep="\t")
        frames.append((de[de.pvalue < 0.01].groupby("cell_type").size()
                       .rename(ds).to_frame()))
    cnt = pd.concat(frames, axis=1).fillna(0).sort_values("GSE209781")
    cnt = cnt.drop(index=[i for i in ("LowQuality", "Unassigned")
                          if i in cnt.index])
    yy = np.arange(len(cnt))
    ax.barh(yy - 0.2, cnt.GSE131882, height=0.38, color="#4C72B0",
            edgecolor="white", linewidth=0.3, label="GSE131882 (snRNA)")
    ax.barh(yy + 0.2, cnt.GSE209781, height=0.38, color="#DD8452",
            edgecolor="white", linewidth=0.3, label="GSE209781 (scRNA)")
    ax.set_yticks(yy)
    ax.set_yticklabels(cnt.index, fontsize=6.6)
    ax.set_xlabel("Genes at P < 0.01", fontweight="bold")
    ax.set_xlim(0, cnt.values.max() * 1.05)
    FS.legend(ax, loc="center right", bbox_to_anchor=(1.0, 0.42))

    ax = fig.add_subplot(gs[2, 2])
    FS.panel(ax, "i")
    nmf = prog[prog.route == "NMF"].drop_duplicates("cell_type")
    nmf = nmf.sort_values("stability")
    ax.barh(range(len(nmf)), nmf.stability, color="#8172B3",
            edgecolor="white", linewidth=0.3)
    for i, r in enumerate(nmf.itertuples()):
        ax.text(r.stability + 0.012, i, f"K={int(r.k_selected)}",
                va="center", fontsize=5.4)
    ax.set_yticks(range(len(nmf)))
    ax.set_yticklabels(nmf.cell_type, fontsize=6.6)
    ax.set_xlabel("Consensus NMF silhouette", fontweight="bold")
    ax.set_xlim(0, float(nmf.stability.max()) * 1.28)

    FS.save(fig, FIG, "Fig2_single_cell_atlas")


# --------------------------------------------------------------------------- #
def figure3() -> None:
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    FS.panel(ax, "a")
    coh = ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1"]
    ranks = {c: pd.read_csv(BULK / f"{c}_rank.tsv.gz", sep="\t", index_col=0)
             for c in coh}
    common = set.intersection(*[set(r.index) for r in ranks.values()])
    common = sorted(common)
    prof = pd.DataFrame({c: ranks[c].loc[common].mean(axis=1) for c in coh})
    ax.scatter(prof[coh[0]], prof[coh[1]], s=1.1, alpha=0.25,
               color="#4C72B0", linewidths=0, rasterized=True)
    r = float(prof[coh[0]].corr(prof[coh[1]], method="spearman"))
    ax.text(0.95, 0.06, f"Spearman ρ = {r:.2f}\n{len(common):,} genes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    ax.set_xlabel("Mean rank, GSE30528", fontweight="bold")
    ax.set_ylabel("Mean rank, GSE96804", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    FS.panel(ax, "b")
    sel = pd.read_csv(TAB / "T12_selected_pairs.tsv.gz", sep="\t")
    lab = {1: "Within program", 2: "Same cell type", 3: "Across cell types"}
    cnt = (sel.groupby(["compartment", "category"]).size()
           .unstack(fill_value=0))
    xx = np.arange(len(cnt))
    bottom = np.zeros(len(cnt))
    for i, c in enumerate(sorted(cnt.columns)):
        ax.bar(xx, cnt[c], bottom=bottom, width=0.55,
               color=FS.PALETTE[i], edgecolor="white", linewidth=0.3,
               label=lab.get(c, str(c)))
        bottom += cnt[c].values
    ax.set_xticks(xx)
    ax.set_xticklabels(["Glomerulus", "Tubulo-\ninterstitium"], fontsize=6.5)
    ax.set_ylabel("Selected pairs (all folds)", fontweight="bold")
    ax.set_ylim(0, bottom.max() * 1.55)
    FS.legend(ax, loc="upper right", handlelength=1.2, handletextpad=0.5)

    ax = fig.add_subplot(gs[0, 2])
    FS.panel(ax, "c")
    data = [sub.delta.abs().values for _, sub in sel.groupby("compartment")]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(FS.PALETTE[i])
        b.set_alpha(0.55)
        b.set_edgecolor("white")
    bp = ax.boxplot(data, widths=0.14, showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor="#444444", linewidth=0.7)
    for el in ("whiskers", "caps", "medians"):
        for ln in bp[el]:
            ln.set(color="#444444", linewidth=0.7)
    ax.axhline(0.25, color="#888888", ls="--", lw=0.8)
    ax.text(0.5, 0.255, "screening threshold", fontsize=5.4, color="#666666",
            va="bottom")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Glomerulus", "Tubulo-\ninterstitium"], fontsize=6.5)
    ax.set_ylabel("|Δ reversal rate|", fontweight="bold")
    ax.set_ylim(0.15, 1.02)

    ax = fig.add_subplot(gs[1, 0])
    FS.panel(ax, "d")
    for i, (comp, sub) in enumerate(sel.groupby("compartment")):
        ax.scatter(sub.reversal_Control, sub.reversal_DKD, s=6, alpha=0.5,
                   color=FS.PALETTE[i], linewidths=0,
                   label="Glomerulus" if comp == "GLOM" else "Tubulointerstitium")
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Reversal rate, controls", fontweight="bold")
    ax.set_ylabel("Reversal rate, DKD", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    FS.legend(ax, loc="center", handletextpad=0.4)

    ax = fig.add_subplot(gs[1, 1:])
    FS.panel(ax, "e")
    core = pd.read_csv(TAB / "T16_core_pairs.tsv.gz", sep="\t")
    core = core[core.fold_consistency >= 2 / 3].copy()
    long = pd.concat([core[["cell_a"]].rename(columns={"cell_a": "cell"}),
                      core[["cell_b"]].rename(columns={"cell_b": "cell"})])
    top = long.cell.value_counts().head(12)[::-1]
    ax.barh(range(len(top)), top.values, color="#C44E52", edgecolor="white",
            linewidth=0.3)
    for i, v in enumerate(top.values):
        ax.text(v + max(top.values) * 0.015, i, str(v), va="center", fontsize=6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=6.5)
    ax.set_xlabel("Appearances in reproducible pairs (≥2/3 folds)",
                  fontweight="bold")
    ax.set_xlim(0, max(top.values) * 1.16)

    FS.save(fig, FIG, "Fig3_programs_and_pairs")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()

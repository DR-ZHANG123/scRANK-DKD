#!/usr/bin/env python
"""Figures 4-7: model performance, baselines and ablations, interpretation, clinical and disease specificity."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import figstyle as FS                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
FIG = ROOT / "figures"

FS.apply_style()
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}
PRIMARY = "scPair_LASSO"


def short(c: str) -> str:
    """Uniform cohort display names. Stripping the compartment prefix alone is not
    enough: GLOM_ERCB1 and TUB_ERCB1 would both become \"ERCB1\" yet denote different cohorts in two side-by-side panels."""
    return FS.cohort(c)


# --------------------------------------------------------------------------- #
def figure4() -> None:
    pred = pd.read_csv(MET / "loco_predictions.tsv.gz", sep="\t")
    res = pd.read_csv(MET / "loco_results.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34)

    for row, comp in enumerate(COMPS):
        sub = pred[(pred.compartment == comp) & (pred.model == PRIMARY)]

        ax = fig.add_subplot(gs[row, 0])
        # Column-major lettering: one plot type per column, one compartment per row, so
        # captions read (a, b) / (c, d) / (e, f) instead of the skipping (a, d) / (b, e) / (c, f)
        FS.panel(ax, "a" if row == 0 else "b")
        for i, (held, g) in enumerate(sub.groupby("held_out")):
            fpr, tpr, _ = roc_curve(g.y, g.p)
            auc = res[(res.compartment == comp) & (res.model == PRIMARY) &
                      (res.held_out == held)].auroc.iloc[0]
            ax.plot(fpr, tpr, color=FS.PALETTE[i], lw=1.3,
                    label=f"{short(held)} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=0.8)
        ax.set_xlabel("1 − specificity", fontweight="bold")
        ax.set_ylabel("Sensitivity", fontweight="bold")
        ax.text(0.98, 0.05, COMPS[comp], transform=ax.transAxes, ha="right",
                fontsize=6.5, style="italic")
        FS.legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.14),
                  handlelength=1.2)

        ax = fig.add_subplot(gs[row, 1])
        FS.panel(ax, "c" if row == 0 else "d")
        for i, (held, g) in enumerate(sub.groupby("held_out")):
            prec, rec, _ = precision_recall_curve(g.y, g.p)
            ap = res[(res.compartment == comp) & (res.model == PRIMARY) &
                     (res.held_out == held)].auprc.iloc[0]
            ax.plot(rec, prec, color=FS.PALETTE[i], lw=1.3,
                    label=f"{short(held)} ({ap:.2f})")
            ax.axhline(g.y.mean(), color=FS.PALETTE[i], ls=":", lw=0.7,
                       alpha=0.6)
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.set_ylim(0, 1.05)
        FS.legend(ax, loc="lower left", handlelength=1.2)

        ax = fig.add_subplot(gs[row, 2])
        FS.panel(ax, "e" if row == 0 else "f")
        r = res[(res.compartment == comp)]
        models = [m for m in FS.MODEL_ORDER if m in set(r.model)]
        ypos = np.arange(len(models))
        for j, m in enumerate(models):
            rows = r[r.model == m]
            mu = rows.auroc.mean()
            lo = rows.auroc_ci_low.mean()
            hi = rows.auroc_ci_high.mean()
            ax.errorbar(mu, j, xerr=[[mu - lo], [hi - mu]], fmt="o",
                        color=("#C44E52" if m == "scPair_LASSO" else "#4C72B0"),
                        markersize=4, capsize=2, lw=1.0)
        ax.axvline(0.5, color="#999999", ls="--", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([FS.disp(m) for m in models], fontsize=6)
        ax.yaxis.tick_right()           # labels on the right, so they cannot overlap the neighbouring panel
        ax.yaxis.set_label_position("right")
        ax.set_xlabel("Macro AUROC (95% CI)", fontweight="bold")
        ax.set_xlim(0.28, 1.04)
        ax.set_ylim(-0.7, len(models) - 0.3)

    FS.save(fig, FIG, "Fig4_model_performance")


# --------------------------------------------------------------------------- #
def figure5() -> None:
    res = pd.read_csv(MET / "loco_results.tsv", sep="\t")
    macro = pd.read_csv(TAB / "T13_macro_performance.tsv", sep="\t")
    # The former panel a (bar chart of macro AUROC per model) was dropped: it restated the
    # AUROC column of Table 1 without confidence intervals, so it carried less than the table.
    # The three remaining panels do not overlap: fold-to-fold agreement (heatmap), between-model differences (forest plot), matched null distribution (negative control).
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.38,
                          height_ratios=[1.0, 0.92])

    # c: per-fold AUROC heatmap, spanning the full row so the cells are large enough
    ax = fig.add_subplot(gs[1, :])
    FS.panel(ax, "c")
    piv = res.pivot_table(index="model", columns=["compartment", "held_out"],
                          values="auroc")
    piv = piv.reindex([m for m in FS.MODEL_ORDER if m in piv.index])
    im = ax.imshow(piv.values, cmap="RdYlBu_r", vmin=0.4, vmax=1.0,
                   aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.8,
                        color="white" if v > 0.85 or v < 0.55 else "#222222")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([short(h) for _, h in piv.columns],
                       rotation=90, fontsize=5.4)
    # Cohort names do not reveal the compartment (GSE30528 is glomerular, GSE30529 is
    # tubulointerstitial), so add a grouping annotation below the axis
    comps_ = [c for c, _ in piv.columns]
    for cname in dict.fromkeys(comps_):
        idx = [i for i, c in enumerate(comps_) if c == cname]
        lo, hi = min(idx), max(idx)
        # Placed above the heatmap: the space below is taken by the vertical cohort labels
        ax.plot([lo - 0.42, hi + 0.42], [1.035, 1.035], color="#444444",
                lw=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((lo + hi) / 2, 1.055, COMPS[cname], ha="center", va="bottom",
                fontsize=6.4, fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([FS.disp(m) for m in piv.index], fontsize=5.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("AUROC", fontsize=5.8)

    # c: difference against the DRGpair baseline
    ax = fig.add_subplot(gs[0, 0])
    FS.panel(ax, "a")
    cmp_ = pd.read_csv(TAB / "T15_model_comparison_pooled.tsv", sep="\t")
    cmp_ = cmp_[cmp_.reference == "DRGpair_LASSO"]
    order = [m for m in FS.MODEL_ORDER if m in set(cmp_.model)]
    yy = np.arange(len(order))
    for i, comp in enumerate(COMPS):
        vals = cmp_[cmp_.compartment == comp].set_index("model").reindex(order)
        ax.errorbar(vals.delta, yy + (i - 0.5) * 0.3,
                    xerr=[vals.delta - vals.lo, vals.hi - vals.delta],
                    fmt="o", color=FS.PALETTE[i], markersize=3.4, capsize=2,
                    lw=0.9, label=COMPS[comp])
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([FS.disp(m) for m in order], fontsize=5.8)
    ax.set_xlabel("ΔAUROC vs genome-wide DRGpair", fontweight="bold")
    # The lower two error bars would sit under a lower-left legend; move the legend to the
    # blank upper left and leave one row of headroom at the top of the y axis
    # A full row of clearance at the top holds the legend without touching any error bar
    ax.set_ylim(-0.6, len(order) + 0.95)
    FS.legend(ax, loc="upper left", handlelength=1.1, ncol=2,
              borderaxespad=0.3)

    # b: negative controls
    ax = fig.add_subplot(gs[0, 1])
    FS.panel(ax, "b")
    try:
        perm = pd.read_csv(MET / "label_permutation.tsv.gz", sep="\t")
        null = pd.read_csv(MET / "random_gene_null.tsv.gz", sep="\t")
    except Exception:
        perm = null = pd.DataFrame()
    if not perm.empty:
        # Only the random-gene null is drawn as a boxplot: the observed value sits on its own
        # box, so \"compared with what\" is immediate. The label-permutation null was exactly
        # 0.5 in every iteration (no pair passed the full screening rule), so drawing it as a
        # box would produce a fake one; it is annotated on the 0.5 reference line instead.
        nc = pd.read_csv(TAB / "T26_negative_controls.tsv", sep="\t")
        nc = nc[nc.model == PRIMARY].set_index(["compartment", "null"])
        data, labels = [], []
        for comp in COMPS:
            data.append(null[null.compartment == comp].macro_auroc.values)
            labels.append("Glomerulus" if comp == "GLOM"
                          else "Tubulointerstitium")
        bp = ax.boxplot(data, patch_artist=True, widths=0.42, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor="#CCB974", alpha=0.6, edgecolor="#444444",
                    linewidth=0.7)
        for el in ("whiskers", "caps", "medians"):
            for ln in bp[el]:
                ln.set(color="#444444", linewidth=0.7)

        for i, comp in enumerate(COMPS):
            obs = macro[(macro.compartment == comp) &
                        (macro.model == PRIMARY)].macro_auroc.iloc[0]
            pv = nc.loc[(comp, "random_gene_sets"), "empirical_p"]
            ax.scatter([i + 1], [obs], marker="D", s=26, color="#C44E52",
                       zorder=6, edgecolors="white", linewidths=0.6,
                       label="Observed scPair-LASSO" if i == 0 else None)
            ax.annotate(f"{obs:.3f}\nP = {pv:.2f}" if pv > 0.05
                        else f"{obs:.3f}\nP = {pv:.3f}",
                        xy=(i + 1, obs), xytext=(i + 1.3, obs + 0.06),
                        fontsize=5.6, color="#C44E52", fontweight="bold",
                        ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color="#C44E52",
                                        lw=0.6, shrinkA=0, shrinkB=2))

        ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
        ax.text(0.5, 0.5, " Label-permutation null: 0.5 in every iteration",
                fontsize=5.4, color="#777777", va="bottom", ha="left")
        ax.set_xticklabels(labels, fontsize=6.6)
        ax.set_xlim(0.45, 2.9)
        ax.set_ylabel("Macro AUROC of\nmatched random gene sets",
                      fontweight="bold")
        ax.set_ylim(0.42, 1.16)
        FS.legend(ax, loc="upper left", handlelength=1.0, handletextpad=0.3)
    FS.save(fig, FIG, "Fig5_baselines_ablation")


# --------------------------------------------------------------------------- #
def figure6() -> None:
    """Figure 6 (restructured): interpretation without attention - pair stability, sensitivity
    to pair count, core pairs and gene recurrence. Attention-by-cell-type moved to the supplement because the cell attribution is unreliable."""
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42)

    sel = pd.read_csv(TAB / "T12_selected_pairs.tsv.gz", sep="\t")

    # a: bootstrap stability distribution of the retained pairs
    ax = fig.add_subplot(gs[0, 0])
    FS.panel(ax, "a")
    for i, (comp, sub) in enumerate(sel.groupby("compartment")):
        ax.hist(sub.stability, bins=np.linspace(0.5, 1.0, 16), alpha=0.6,
                color=FS.PALETTE[i], edgecolor="white", linewidth=0.3,
                label=COMPS[comp])
    ax.set_xlabel("Bootstrap selection frequency", fontweight="bold")
    ax.set_ylabel("Selected pairs", fontweight="bold")
    FS.legend(ax, loc="upper left", handlelength=1.1)

    # b: core pairs, reproduced across folds, with direction
    ax = fig.add_subplot(gs[0, 1])
    FS.panel(ax, "b")
    core = pd.read_csv(TAB / "T16_core_pairs.tsv.gz", sep="\t")
    top = (core[core.fold_consistency >= 2 / 3]
           .assign(absd=lambda d: d.mean_delta.abs())
           .sort_values("absd", ascending=False).head(14)[::-1])
    yy = np.arange(len(top))
    ax.barh(yy, top.mean_delta,
            color=["#C44E52" if v > 0 else "#4C72B0" for v in top.mean_delta],
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{a} > {b}" for a, b in zip(top.gene_a, top.gene_b)],
                       fontsize=5.2)
    ax.set_xlabel("Δ reversal rate (DKD − control)", fontweight="bold")
    lim = float(top.mean_delta.abs().max()) * 1.25
    ax.set_xlim(-lim, lim)

    # c: gene recurrence (in how many folds a gene appears among the selected core pairs)
    ax = fig.add_subplot(gs[1, 0])
    FS.panel(ax, "c")
    try:
        g = pd.read_csv(TAB / "T16b_core_genes.tsv", sep="\t")
    except Exception:
        g = pd.DataFrame()
    if not g.empty:
        sub = (g.sort_values(["max_folds", "n_pairs"], ascending=False)
               .groupby("compartment").head(9))
        # Take the top genes of each compartment and draw horizontal bars, shaded by max_folds
        gg = sub.sort_values("n_pairs").tail(16)
        yy = np.arange(len(gg))
        colors = [FS.PALETTE[0] if c == "GLOM" else FS.PALETTE[1]
                  for c in gg.compartment]
        ax.barh(yy, gg.n_pairs, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{r.gene}" for r in gg.itertuples()], fontsize=5.0)
        ax.set_xlabel("Reproducible pairs containing the gene",
                      fontweight="bold")
        import matplotlib.patches as mpatches
        FS.legend(ax, handles=[mpatches.Patch(color=FS.PALETTE[0], label="Glom"),
                               mpatches.Patch(color=FS.PALETTE[1], label="Tub")],
                  loc="lower right")

    # d: sensitivity to the number of pairs
    ax = fig.add_subplot(gs[1, 1])
    FS.panel(ax, "d")
    try:
        ns = pd.read_csv(TAB / "T22_pair_count_sensitivity.tsv", sep="\t")
    except Exception:
        ns = pd.DataFrame()
    if not ns.empty:
        for i, comp in enumerate(COMPS):
            gp = ns[ns.compartment == comp].groupby("n_pairs").auroc
            m, sd = gp.mean(), gp.std()
            ax.errorbar(m.index, m.values, yerr=sd.values, marker="o",
                        markersize=3.5, capsize=2, lw=1.1,
                        color=FS.PALETTE[i], label=COMPS[comp])
        ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200, 300])
        ax.set_xticklabels([20, 50, 100, 200, 300])
        ax.set_xlabel("Number of gene pairs in model", fontweight="bold")
        ax.set_ylabel("External AUROC", fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        FS.legend(ax, loc="lower right")
    FS.save(fig, FIG, "Fig6_interpretation")


# --------------------------------------------------------------------------- #
def figure7() -> None:
    sc_ = pd.read_csv(MET / "specificity_scores.tsv.gz", sep="\t")
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.36)

    order = ["Control", "TMD", "MCD", "FSGS_MCD", "MGN", "IgAN", "LN", "HTN",
             "FSGS", "RPGN", "DKD"]
    for i, comp in enumerate(COMPS):
        ax = fig.add_subplot(gs[0, i])
        FS.panel(ax, "ab"[i])
        sub = sc_[sc_.compartment == comp]
        groups = [g for g in order if (sub.group == g).sum() >= 3]
        data = [sub[sub.group == g].score.values for g in groups]
        colors = ["#4C72B0" if g == "Control" else
                  ("#C44E52" if g == "DKD" else "#BBBBBB") for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        # When scores saturate the boxes are nearly flat and fill colour alone is unreadable;
        # outline Control and DKD in the same colour and thicken the median so flat boxes still read
        for box, col, g in zip(bp["boxes"], colors, groups):
            hi = g in ("Control", "DKD")
            box.set(facecolor=col, alpha=0.68,
                    edgecolor=col if hi else "#444444",
                    linewidth=1.4 if hi else 0.6)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set(color="#444444", linewidth=0.6)
        for ln, col, g in zip(bp["medians"], colors, groups):
            hi = g in ("Control", "DKD")
            ln.set(color=col if hi else "#444444", linewidth=1.8 if hi else 0.6)
        rng = np.random.default_rng(7)
        for j, v in enumerate(data):
            ax.scatter(rng.normal(j + 1, 0.06, len(v)), v, s=3.2, alpha=0.55,
                       color="#333333", linewidths=0, zorder=4)
        ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(groups, data)],
                           fontsize=5, rotation=90)
        ax.set_ylabel("scPair-LASSO score", fontweight="bold")
        ax.set_title(COMPS[comp], fontsize=7, pad=2)
        ax.set_ylim(-0.05, 1.14)

    # c/d: relationship with surrogate measures of kidney function
    for i, (col, label) in enumerate((("EGF_rank", "Intrarenal EGF (rank)"),
                                      ("fibrosis_score",
                                       "Fibrosis module score"))):
        ax = fig.add_subplot(gs[1, i])
        FS.panel(ax, "cd"[i])
        clin = pd.read_csv(TAB / "T20_surrogate_clinical.tsv", sep="\t")
        for j, comp in enumerate(COMPS):
            sub = sc_[sc_.compartment == comp].dropna(subset=[col, "score"])
            ax.scatter(sub.score, sub[col], s=5, alpha=0.5,
                       color=FS.PALETTE[j], linewidths=0)
            row = clin[(clin.compartment == comp) & (clin.variable == col) &
                       (clin.scope == "all_samples")]
            if len(row):
                z = np.polyfit(sub.score, sub[col], 1)
                xs = np.linspace(0, 1, 50)
                ax.plot(xs, np.polyval(z, xs), color=FS.PALETTE[j], lw=1.2,
                        label=f"{COMPS[comp]}: ρ={row.spearman_rho.iloc[0]:.2f}")
        ax.set_xlabel("scPair-LASSO score", fontweight="bold")
        ax.set_ylabel(label, fontweight="bold")
        FS.legend(ax, loc="lower left" if col == "EGF_rank" else "lower right",
                  handlelength=1.1)

    FS.save(fig, FIG, "Fig7_clinical_specificity")


if __name__ == "__main__":
    figure4()
    figure5()
    figure6()
    figure7()

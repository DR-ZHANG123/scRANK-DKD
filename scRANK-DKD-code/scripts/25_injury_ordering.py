#!/usr/bin/env python
"""Cross-platform directed injury ordering: the study's main biological output.

Rather than maximising classification AUC, this produces an ordering of DKD injury
genes that uses no cell labels and reproduces across platforms, and tests whether
it forms a monotone injury continuum.

1. Each candidate gene's DKD-versus-control rank shift within each bulk cohort is
   combined by random-effects meta-analysis, giving a pooled effect, 95% CI and I^2.
2. The meta-significant rising and falling genes define a patient-level injury
   continuum score (mean percentile rank of rising genes minus that of falling
   genes), containing no cell label.
3. The continuum is tested for monotone association: control < other CKD < DKD, and
   against the fibrosis module after removing genes shared with the continuum.
4. Cross-cohort consistency of the effects is reported (I^2, number of cohorts
   agreeing in direction).

Writes T39 (per-gene meta-analysis), T40 (continuum associations) and Fig13."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import data as D                                # noqa: E402
from scdrp import figstyle as FS                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
PAIR = ROOT / "data_processed" / "pair_matrix"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"

FS.apply_style()
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}
SPEC = {"GLOM": ["GLOM_ERCB1", "GLOM_ERCB2"],
        "TUB": ["TUB_ERCB1", "TUB_ERCB2"]}
FIBROSIS = ["COL1A1", "COL1A2", "COL3A1", "FN1", "TIMP1", "ACTA2",
            "POSTN", "LUM", "VIM"]
CKD_ORDER = ["Control", "MCD", "TMD", "MGN", "IgAN", "LN", "HTN",
             "FSGS", "FSGS_MCD", "RPGN", "DKD"]


def gene_rank_shift(cid: str, genes: list[str]) -> pd.Series:
    """Per-cohort DKD-versus-control shift in within-sample percentile rank, per gene."""
    rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
    meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
    grp = meta["group"].reindex(rank.columns)
    dkd = rank.loc[[g for g in genes if g in rank.index], grp[grp == "DKD"].index]
    ctl = rank.loc[[g for g in genes if g in rank.index], grp[grp == "Control"].index]
    return (dkd.mean(axis=1) - ctl.mean(axis=1))


def se_of_shift(cid: str, genes: list[str]) -> pd.Series:
    """Standard error of the rank shift, from the two group variances."""
    rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
    meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
    grp = meta["group"].reindex(rank.columns)
    gg = [g for g in genes if g in rank.index]
    d = rank.loc[gg, grp[grp == "DKD"].index]
    c = rank.loc[gg, grp[grp == "Control"].index]
    nd, nc = d.shape[1], c.shape[1]
    return np.sqrt(d.var(axis=1) / max(nd, 1) + c.var(axis=1) / max(nc, 1))


def dl_meta_gene(shifts: pd.DataFrame, ses: pd.DataFrame) -> pd.DataFrame:
    """DerSimonian-Laird random-effects pooling of one gene's shift across cohorts."""
    rows = []
    for g in shifts.index:
        y = shifts.loc[g].values.astype(float)
        s = ses.loc[g].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
        y, s = y[ok], s[ok]
        k = len(y)
        if k < 2:
            continue
        w = 1 / s ** 2
        fixed = (w * y).sum() / w.sum()
        Q = (w * (y - fixed) ** 2).sum()
        C = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
        wr = 1 / (s ** 2 + tau2)
        pooled = (wr * y).sum() / wr.sum()
        se = np.sqrt(1 / wr.sum())
        i2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
        z = pooled / se
        from scipy.stats import norm
        rows.append(dict(gene=g, meta_shift=pooled, se=se,
                         ci_low=pooled - 1.96 * se, ci_high=pooled + 1.96 * se,
                         i2=i2, n_cohorts=k,
                         n_same_direction=int((np.sign(y) == np.sign(pooled)).sum()),
                         z=z, p=float(2 * norm.sf(abs(z)))))
    out = pd.DataFrame(rows)
    if not out.empty:
        from statsmodels.stats.multitest import multipletests
        out["fdr"] = multipletests(out.p, method="fdr_bh")[1]
    return out.sort_values("meta_shift")


def continuum_score(rank: pd.DataFrame, up: list[str], down: list[str]
                    ) -> pd.Series:
    up_ = [g for g in up if g in rank.index]
    dn_ = [g for g in down if g in rank.index]
    return rank.reindex(up_).mean(axis=0) - rank.reindex(dn_).mean(axis=0)


def main() -> None:
    gene_meta_all, cont_rows = [], []
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    fig.subplots_adjust(hspace=0.40, wspace=0.40, left=0.12,
                        right=0.97, top=0.94, bottom=0.11)

    for row, comp_name in enumerate(COMPS):
        comp = D.load_compartment(comp_name)
        genes = list(comp.genes)
        cohorts = list(comp.cohorts)

        shifts = pd.DataFrame({c: gene_rank_shift(c, genes) for c in cohorts})
        ses = pd.DataFrame({c: se_of_shift(c, genes) for c in cohorts})
        gm = dl_meta_gene(shifts, ses)
        gm["compartment"] = comp_name
        gene_meta_all.append(gm)

        sig = gm[gm.fdr < 0.05]
        up = list(sig[sig.meta_shift > 0].gene)
        down = list(sig[sig.meta_shift < 0].gene)
        print(f"[{comp_name}] meta-significant: {len(up)} up, {len(down)} down "
              f"(FDR<0.05); median I^2 = {gm.i2.median():.0f}%")

        up_c = [g for g in up if g not in FIBROSIS]
        down_c = [g for g in down if g not in FIBROSIS]
        recs = []
        for cid in SPEC[comp_name]:
            rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
            meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
            score = continuum_score(rank, up_c, down_c)
            fib = rank.reindex([g for g in FIBROSIS if g in rank.index]).mean(axis=0)
            df = pd.DataFrame(dict(sample=score.index, score=score.values,
                                   fibrosis=fib.reindex(score.index).values))
            df["group"] = meta["group"].reindex(score.index).values
            df["cohort"] = cid
            recs.append(df)
        cont = pd.concat(recs, ignore_index=True)
        cont["compartment"] = comp_name
        cont_rows.append(cont)

        v = cont.dropna(subset=["score", "fibrosis"])
        rho, p = spearmanr(v.score, v.fibrosis)
        g_ctrl = cont[cont.group == "Control"].score
        g_ckd = cont[~cont.group.isin(["Control", "DKD"])].score
        g_dkd = cont[cont.group == "DKD"].score
        from scipy.stats import kruskal
        kr = kruskal(g_ctrl, g_ckd, g_dkd) if min(len(g_ctrl), len(g_ckd),
                                                  len(g_dkd)) >= 2 else None
        cont_rows_summary = dict(
            compartment=comp_name, n_up=len(up_c), n_down=len(down_c),
            spearman_fibrosis=round(float(rho), 3), p_fibrosis=float(p),
            median_control=round(float(g_ctrl.median()), 3),
            median_otherCKD=round(float(g_ckd.median()), 3),
            median_DKD=round(float(g_dkd.median()), 3),
            kruskal_p=float(kr.pvalue) if kr else np.nan)
        cont_rows.append(pd.DataFrame([cont_rows_summary]))

        ax = axes[row, 0]
        FS.panel(ax, "ab"[row])
        pick = pd.concat([gm.head(8), gm.tail(8)])
        yy = np.arange(len(pick))
        colors = ["#4C72B0" if v < 0 else "#C44E52" for v in pick.meta_shift]
        ax.errorbar(pick.meta_shift, yy,
                    xerr=[pick.meta_shift - pick.ci_low,
                          pick.ci_high - pick.meta_shift],
                    fmt="o", ecolor="#999999", markersize=0, lw=0.8, zorder=1)
        ax.scatter(pick.meta_shift, yy, c=colors, s=16, zorder=2,
                   edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels(pick.gene, fontsize=5.2)
        ax.set_xlabel("Meta rank shift in DKD (95% CI)", fontweight="bold")
        ax.set_title(COMPS[comp_name], fontsize=7, pad=3)

        ax = axes[row, 1]
        FS.panel(ax, "cd"[row])
        groups = [g for g in CKD_ORDER if (cont.group == g).sum() >= 3]
        data = [cont[cont.group == g].score.values for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for box, g in zip(bp["boxes"], groups):
            col = "#4C72B0" if g == "Control" else ("#C44E52" if g == "DKD"
                                                    else "#BBBBBB")
            box.set(facecolor=col, alpha=0.7, edgecolor="#444444", linewidth=0.6)
        for el in ("whiskers", "caps", "medians"):
            for ln in bp[el]:
                ln.set(color="#444444", linewidth=0.6)
        ax.set_xticklabels(groups, rotation=90, fontsize=5.2)
        ax.set_ylabel("Injury continuum score", fontweight="bold")
        ax.set_title(COMPS[comp_name], fontsize=7, pad=3)
        kp = float(kr.pvalue) if kr else float("nan")
        exp = int(np.floor(np.log10(kp))) if kp == kp and kp > 0 else 0
        mant = kp / 10 ** exp
        ax.text(0.03, 0.97,
                f"vs fibrosis $\\rho$={rho:.2f}\n"
                f"Kruskal $P$={mant:.1f}$\\times$10$^{{{exp}}}$",
                transform=ax.transAxes, va="top", fontsize=5.8)

    FS.save(fig, FIG, "Fig13_injury_ordering")

    gm_all = pd.concat(gene_meta_all, ignore_index=True)
    gm_all.to_csv(TAB / "T39_gene_shift_meta.tsv.gz", sep="\t", index=False)
    cont_summary = pd.concat([r for r in cont_rows if "n_up" in r.columns],
                             ignore_index=True)
    cont_summary.to_csv(TAB / "T40_injury_continuum.tsv", sep="\t", index=False)
    print("\n===== injury continuum (fibrosis-overlapping genes removed) =====")
    print(cont_summary.to_string(index=False))


if __name__ == "__main__":
    main()

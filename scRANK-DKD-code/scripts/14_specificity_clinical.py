#!/usr/bin/env python
"""Disease specificity and surrogate clinical association.

None of the public cohorts included carries eGFR, UACR or creatinine (see T01), so
clinical association uses two validated intrarenal surrogates:
  - intrarenal EGF mRNA, a tissue correlate of eGFR and of CKD progression;
  - a fibrosis module score, a transcriptomic surrogate for interstitial fibrosis.
Both are computed on within-sample percentile ranks, the same representation the model uses.

Disease specificity: models trained on DKD versus control are applied to other CKD samples
（IgAN / FSGS / MGN / LN / HTN / MCD / RPGN / TMD），
to split the score into a shared kidney-injury part and a DKD-enriched part.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                        # noqa: E402
from scdrp import data as D                              # noqa: E402
from scdrp import screening as SC                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
PAIR = ROOT / "data_processed" / "pair_matrix"
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"

SEED = 20260722
N_PAIRS_FINAL = 100

# cohorts used for specificity assessment (they contain other-CKD samples)
SPEC_COHORTS = {"GLOM": ["GLOM_ERCB1", "GLOM_ERCB2"],
                "TUB": ["TUB_ERCB1", "TUB_ERCB2"]}
# cohorts that took no part in any training
FULLY_HELD_OUT = {"GLOM_ERCB2"}

FIBROSIS = ["COL1A1", "COL1A2", "COL3A1", "FN1", "TIMP1", "ACTA2",
            "POSTN", "LUM", "VIM"]
EGF_SURROGATE = ["EGF"]

# EGF and several fibrosis genes are themselves in the candidate pool, so the raw
# correlation is partly self-correlation. Both versions are reported: all surrogate genes, and the version with candidate-pool genes removed.


def fit_model(comp: D.Compartment, exclude: str | None) -> tuple[np.ndarray, object]:
    """Run screening on the given training cohorts and fit scPair-LASSO.

    exclude names the cohort to leave out (that cohort is then scored as a whole), so its
    DKD, control and other-CKD samples all sit outside the model and share one scoring
    scale. With exclude=None the final model is trained on all LOCO cohorts.
    """
    cds = [c for k, c in comp.cohorts.items() if k != exclude]
    P, M, mask, y, cid = D.stack(cds)
    sel = SC.select_pairs(P, M, mask, y, cid, comp.pairs,
                          n_max=N_PAIRS_FINAL, seed=SEED)
    pair_idx = sel.pair_id.values
    X = np.hstack([np.where(mask[pair_idx], P[pair_idx], 0.5).T,
                   np.where(mask[pair_idx], M[pair_idx], 0.0).T])
    model = bl.fit_lasso_cv(X, y)
    tag = exclude or "all LOCO cohorts"
    print(f"[{comp.name}] training (excluding {tag}): {len(pair_idx)} gene pairs, "
          f"n={len(y)}，DKD {int(y.sum())}")
    return pair_idx, model


def score_cohort(comp: D.Compartment, cid: str, pair_idx: np.ndarray, model):
    npz = np.load(PAIR / f"{comp.name}_pairs.npz", allow_pickle=True)
    samples = np.asarray(npz[f"{cid}__samples"], dtype=object)
    P = npz[f"{cid}__P"][pair_idx]
    M = npz[f"{cid}__M"][pair_idx]
    mask = npz[f"{cid}__mask"][pair_idx]
    X = np.hstack([np.where(mask, P, 0.5).T, np.where(mask, M, 0.0).T])
    return samples, model.predict_proba(X)[:, 1]


def candidate_genes(comp_name: str) -> set[str]:
    return set(pd.read_csv(PAIR / f"{comp_name}_candidate_genes.tsv",
                           sep="\t")["gene"])


def surrogates(cid: str, samples, in_model: set[str]) -> pd.DataFrame:
    rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
    rank = rank[list(samples)]
    egf = rank.reindex(EGF_SURROGATE).mean(axis=0)
    fib_all = [g for g in FIBROSIS if g in rank.index]
    fib_ind = [g for g in fib_all if g not in in_model]
    out = {"EGF_rank": egf.values,
           "EGF_in_model": np.repeat(bool(set(EGF_SURROGATE) & in_model),
                                     len(samples)),
           "fibrosis_score": rank.reindex(fib_all).mean(axis=0).values,
           "fibrosis_score_independent":
               (rank.reindex(fib_ind).mean(axis=0).values if fib_ind
                else np.full(len(samples), np.nan)),
           "n_fibrosis_genes_independent": np.repeat(len(fib_ind),
                                                     len(samples))}
    return pd.DataFrame(out, index=samples)


def main() -> None:
    all_scores, spec_rows, clin_rows = [], [], []

    for comp_name in ("GLOM", "TUB"):
        comp = D.load_compartment(comp_name)
        for cid in SPEC_COHORTS[comp_name]:
            # If the cohort took part in LOCO, use the fold that held it out;
            # a cohort that never entered training is scored by the final all-cohort model.
            exclude = cid if cid in comp.cohorts else None
            pair_idx, model = fit_model(comp, exclude)
            meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
            samples, p = score_cohort(comp, cid, pair_idx, model)
            sur = surrogates(cid, samples, candidate_genes(comp_name))
            df = pd.DataFrame(dict(compartment=comp_name, cohort=cid,
                                   sample=samples, score=p))
            df["group"] = meta["group"].reindex(samples).values
            df = df.join(sur.reset_index(drop=True))
            df["fully_held_out"] = cid in FULLY_HELD_OUT
            df["scored_by"] = ("held-out fold" if exclude
                               else "final model (cohort not used in training)")
            all_scores.append(df)

        scores = pd.concat(all_scores[-len(SPEC_COHORTS[comp_name]):],
                           ignore_index=True)

        # ---- disease specificity: DKD vs other CKD vs control ----
        ctl = scores[scores.group == "Control"].score.values
        dkd = scores[scores.group == "DKD"].score.values
        for grp, sub in scores.groupby("group"):
            v = sub.score.values
            row = dict(compartment=comp_name, group=grp, n=len(v),
                       median_score=round(float(np.median(v)), 4))
            if grp != "Control" and len(ctl) >= 3 and len(v) >= 3:
                row["p_vs_Control"] = float(
                    mannwhitneyu(v, ctl, alternative="greater").pvalue)
            if grp not in ("DKD",) and len(dkd) >= 3 and len(v) >= 3:
                row["p_vs_DKD"] = float(
                    mannwhitneyu(dkd, v, alternative="greater").pvalue)
            spec_rows.append(row)

        # ---- surrogate clinical association ----
        for label, col in (("EGF_rank", "EGF_rank"),
                           ("fibrosis_score", "fibrosis_score"),
                           ("fibrosis_score_independent",
                            "fibrosis_score_independent")):
            for scope, sub in (("all_samples", scores),
                               ("DKD_and_Control",
                                scores[scores.group.isin(["DKD", "Control"])])):
                v = sub.dropna(subset=["score", col])
                if len(v) < 10:
                    continue
                rho, p = spearmanr(v.score, v[col])
                in_model = (set(EGF_SURROGATE) & candidate_genes(comp_name)
                            if col.startswith("EGF")
                            else (set(FIBROSIS) & candidate_genes(comp_name)
                                  if col == "fibrosis_score" else set()))
                clin_rows.append(dict(compartment=comp_name, variable=label,
                                      scope=scope, n=len(v),
                                      spearman_rho=round(float(rho), 3),
                                      p_value=float(p),
                                      surrogate_genes_in_model=
                                      ",".join(sorted(in_model)) or "none"))

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(MET / "specificity_scores.tsv.gz", sep="\t", index=False)

    spec = pd.DataFrame(spec_rows)
    # BH and Bonferroni correction over all pairwise comparisons within each compartment:
    # uncorrected P values would call marginal differences across 8 comparator diseases significant
    from statsmodels.stats.multitest import multipletests
    for col in ("p_vs_Control", "p_vs_DKD"):
        spec[col + "_bh"] = np.nan
        spec[col + "_bonf"] = np.nan
        for comp_name, idx in spec.groupby("compartment").groups.items():
            sub = spec.loc[idx, col].dropna()
            if sub.empty:
                continue
            spec.loc[sub.index, col + "_bh"] = multipletests(
                sub.values, method="fdr_bh")[1]
            spec.loc[sub.index, col + "_bonf"] = np.minimum(
                1.0, sub.values * len(sub))
    spec.to_csv(TAB / "T19_disease_specificity.tsv", sep="\t", index=False)
    print("\n===== disease specificity (scPair-LASSO score) =====")
    print(spec.to_string(index=False))

    clin = pd.DataFrame(clin_rows)
    clin.to_csv(TAB / "T20_surrogate_clinical.tsv", sep="\t", index=False)
    print("\n===== correlation with intrarenal surrogates =====")
    print(clin.to_string(index=False))

    # ---- shared injury vs DKD enrichment: per-pair reversal rate in each disease ----
    rows = []
    for comp_name in ("GLOM", "TUB"):
        comp = D.load_compartment(comp_name)
        npz = np.load(PAIR / f"{comp_name}_pairs.npz", allow_pickle=True)
        core = pd.read_csv(TAB / "T16_core_pairs.tsv.gz", sep="\t")
        core = core[(core.compartment == comp_name) &
                    (core.fold_consistency >= 2 / 3)]
        idx = core.pair_id.values
        for cid in SPEC_COHORTS[comp_name]:
            meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
            samples = np.asarray(npz[f"{cid}__samples"], dtype=object)
            P = npz[f"{cid}__P"][idx]
            mask = npz[f"{cid}__mask"][idx]
            grp = meta["group"].reindex(samples).values
            for g in pd.unique(grp):
                sel_g = grp == g
                if sel_g.sum() < 3:
                    continue
                rate = np.where(mask[:, sel_g], P[:, sel_g] == 1, np.nan)
                rows.append(pd.DataFrame(dict(
                    compartment=comp_name, cohort=cid, group=g,
                    pair_id=idx, n=int(sel_g.sum()),
                    reversal_rate=np.nanmean(rate, axis=1))))
    rev = pd.concat(rows, ignore_index=True)
    rev.to_csv(TAB / "T21_pair_reversal_by_disease.tsv.gz", sep="\t", index=False)

    piv = (rev.groupby(["compartment", "group"]).reversal_rate.mean()
           .reset_index())
    print("\n===== mean reversal rate of core pairs in each disease =====")
    print(piv.to_string(index=False))


if __name__ == "__main__":
    main()

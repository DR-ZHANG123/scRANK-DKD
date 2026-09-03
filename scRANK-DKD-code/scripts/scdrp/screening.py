"""Statistical screening of gene pairs and bootstrap stability selection.

Every function sees only the current outer training set and never the held-out cohort.

Two changes from the first version directly affect the validity of the inference:

1. The Fisher test is computed for every pair meeting the coverage requirement, and
   BH correction is applied over that complete family. The first version applied BH
   only to pairs already passing an effect-size and sign-consistency prefilter, but
   that prefilter and the Fisher P value come from the same 2x2 table and are not independent, so the reported FDR was not the quantity claimed. Effect size and sign consistency are now imposed separately, as independent inclusion criteria.

2. Bootstrap stability re-evaluates the entire screening rule (effect size + per-cohort
   sign consistency + nominal significance) rather than recomputing only the effect-size
   condition that selected the candidates in the first place. The first version was
   circular: candidates were chosen by |delta| >= 0.25, then the same condition was used
   to count how often they recurred, so the frequency was necessarily near 1 (637/700 were exactly 1.000) and the 0.5 threshold never bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

DELTA_MIN = 0.25        # minimum difference in reversal rate between DKD and control
FDR_MAX = 0.05
MARGIN_MIN = 0.02       # minimum median rank margin; excludes near-tied pairs
MAX_PAIRS_PER_GENE = 8  # redundancy cap: how many pairs one gene may enter
BOOT_ITERS = 500
BOOT_FREQ_MIN = 0.5
BOOT_P_MAX = 0.05       # nominal significance required within a bootstrap resample


def _counts(P: np.ndarray, mask: np.ndarray,
            y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the 2x2 counts (a, b, c, d): rows are disease status, columns pair direction."""
    pos, neg = y == 1, y == 0
    a = ((P[:, pos] == 1) & mask[:, pos]).sum(axis=1)
    b = ((P[:, pos] == 0) & mask[:, pos]).sum(axis=1)
    c = ((P[:, neg] == 1) & mask[:, neg]).sum(axis=1)
    d = ((P[:, neg] == 0) & mask[:, neg]).sum(axis=1)
    return a, b, c, d


def _reversal_rate(P: np.ndarray, mask: np.ndarray,
                   y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the fraction with P=1 in DKD and in controls, each over available samples."""
    a, b, c, d = _counts(P, mask, y)
    n1, n0 = a + b, c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(n1 > 0, a / np.maximum(n1, 1), np.nan),
                np.where(n0 > 0, c / np.maximum(n0, 1), np.nan))


def _chi2_pvalues(a, b, c, d) -> np.ndarray:
    """Vectorised approximate P values, used for BH over the complete family; retained pairs are rechecked with an exact test."""
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * (a * d - b * c) ** 2
        den = (a + b) * (c + d) * (a + c) * (b + d)
        stat = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    from scipy.stats import chi2
    return chi2.sf(stat, df=1)


def screen_pairs(P: np.ndarray, M: np.ndarray, mask: np.ndarray,
                 y: np.ndarray, cohort_id: np.ndarray,
                 min_coverage: float = 0.9) -> pd.DataFrame:
    """Univariate screening on the pooled training cohorts, requiring the same direction in every training cohort."""
    coverage = mask.mean(axis=1)
    r_dkd, r_ctl = _reversal_rate(P, mask, y)
    delta = r_dkd - r_ctl
    med_margin = np.nanmedian(np.where(mask, M, np.nan), axis=1)

    # per-cohort sign consistency
    per_cohort = []
    for c in np.unique(cohort_id):
        sel = cohort_id == c
        aa, bb = _reversal_rate(P[:, sel], mask[:, sel], y[sel])
        per_cohort.append(aa - bb)
    per_cohort = np.vstack(per_cohort)
    with np.errstate(invalid="ignore"):
        consistent = (np.sign(per_cohort) == np.sign(delta)[None, :]).all(axis=0)
        min_abs_cohort_delta = np.nanmin(np.abs(per_cohort), axis=0)

    # P values and BH correction over the complete family, unaffected by any effect-size prefilter
    testable = coverage >= min_coverage
    a, b, c_, d = _counts(P, mask, y)
    pvals = np.ones(P.shape[0])
    pvals[testable] = _chi2_pvalues(a[testable], b[testable],
                                    c_[testable], d[testable])
    qvals = np.ones_like(pvals)
    if testable.any():
        qvals[testable] = multipletests(pvals[testable], method="fdr_bh")[1]

    # Retained pairs are rechecked by Fisher's exact test; the chi-square approximation can be anti-conservative at these sample sizes
    passed = (testable & (np.abs(delta) >= DELTA_MIN) & consistent &
              (med_margin >= MARGIN_MIN) & (qvals < FDR_MAX))
    exact_p = np.full(P.shape[0], np.nan)
    for k in np.flatnonzero(passed):
        exact_p[k] = fisher_exact([[a[k], b[k]], [c_[k], d[k]]])[1]
    passed &= np.nan_to_num(exact_p, nan=1.0) < 0.05

    return pd.DataFrame(dict(
        pair_id=np.arange(P.shape[0]), coverage=coverage,
        reversal_DKD=r_dkd, reversal_Control=r_ctl, delta=delta,
        median_margin=med_margin, direction_consistent=consistent,
        min_cohort_delta=min_abs_cohort_delta,
        pvalue=pvals, qvalue=qvals, fisher_p=exact_p,
        passed=passed))


def bootstrap_stability(P: np.ndarray, M: np.ndarray, mask: np.ndarray,
                        y: np.ndarray, cohort_id: np.ndarray,
                        candidates: np.ndarray, n_iter: int = BOOT_ITERS,
                        seed: int = 20260722) -> np.ndarray:
    """Patient-level resampling with replacement, recording how often a candidate pair passes the entire rule again.

    The rule matches the primary screening: effect size + per-cohort sign consistency +
    nominal significance. The frequency therefore reflects whether the whole screening
    cascade reproduces under resampling, not whether the single condition that originally selected the pair still holds.
    """
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(candidates))
    Pc, Mc, mc = P[candidates], M[candidates], mask[candidates]
    for _ in range(n_iter):
        idx = np.concatenate([
            rng.choice(np.flatnonzero(cohort_id == c),
                       size=int((cohort_id == c).sum()), replace=True)
            for c in np.unique(cohort_id)])
        yb, cb = y[idx], cohort_id[idx]
        if len(np.unique(yb)) < 2:
            continue
        Pb, Mb, kb = Pc[:, idx], Mc[:, idx], mc[:, idx]
        a, b, c_, d = _counts(Pb, kb, yb)
        r1, r0 = _reversal_rate(Pb, kb, yb)
        delta = r1 - r0
        ok = np.abs(delta) >= DELTA_MIN
        for c in np.unique(cb):
            sel = cb == c
            aa, bb = _reversal_rate(Pb[:, sel], kb[:, sel], yb[sel])
            with np.errstate(invalid="ignore"):
                ok &= np.sign(aa - bb) == np.sign(delta)
        ok &= _chi2_pvalues(a, b, c_, d) < BOOT_P_MAX
        ok &= np.nanmedian(np.where(kb, Mb, np.nan), axis=1) >= MARGIN_MIN
        hits += np.nan_to_num(ok.astype(float))
    return hits / n_iter


def prune_redundant(stats: pd.DataFrame, pairs: pd.DataFrame,
                    max_per_gene: int = MAX_PAIRS_PER_GENE) -> pd.DataFrame:
    """Greedy retention in descending stability, capping how often one gene recurs."""
    merged = stats.merge(pairs[["pair_id", "gene_a", "gene_b", "category"]],
                         on="pair_id")
    merged = (merged.assign(_absdelta=merged.delta.abs())
              .sort_values(["stability", "_absdelta", "pair_id"],
                           ascending=[False, False, True], kind="mergesort")
              .drop(columns="_absdelta"))
    used: dict[str, int] = {}
    keep = []
    for _, r in merged.iterrows():
        if used.get(r.gene_a, 0) >= max_per_gene or \
           used.get(r.gene_b, 0) >= max_per_gene:
            continue
        used[r.gene_a] = used.get(r.gene_a, 0) + 1
        used[r.gene_b] = used.get(r.gene_b, 0) + 1
        keep.append(r.pair_id)
    return merged[merged.pair_id.isin(keep)].reset_index(drop=True)


def select_pairs(P, M, mask, y, cohort_id, pairs: pd.DataFrame,
                 n_max: int = 300, seed: int = 20260722) -> pd.DataFrame:
    """Full screening cascade: univariate -> bootstrap stability -> redundancy pruning."""
    stats = screen_pairs(P, M, mask, y, cohort_id)
    cand = np.flatnonzero(stats.passed.values)
    if cand.size == 0:
        return pd.DataFrame(columns=["pair_id", "stability", "delta"])
    freq = bootstrap_stability(P, M, mask, y, cohort_id, cand, seed=seed)
    sub = stats.iloc[cand].copy()
    sub["stability"] = freq
    kept = sub[sub.stability >= BOOT_FREQ_MIN]
    if kept.empty:                      # when nothing qualifies, keep the most stable few and record it
        kept = sub.nlargest(min(n_max, len(sub)), "stability")
    return prune_redundant(kept, pairs).head(n_max)

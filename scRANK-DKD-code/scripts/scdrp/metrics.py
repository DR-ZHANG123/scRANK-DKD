"""Discrimination, calibration and bootstrap inference."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             roc_auc_score)


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration slope and intercept from a logistic fit of the outcome on the predicted logit."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) < 2 or np.std(lp) < 1e-9:
        return float("nan"), float("nan")
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    yhat = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    slope, intercept = calibration(y, p)
    return dict(
        n=len(y), n_pos=int(y.sum()),
        auroc=float(roc_auc_score(y, p)) if both else float("nan"),
        auprc=float(average_precision_score(y, p)) if both else float("nan"),
        balanced_accuracy=float(balanced_accuracy_score(y, yhat)) if both else float("nan"),
        sensitivity=tp / (tp + fn) if tp + fn else float("nan"),
        specificity=tn / (tn + fp) if tn + fp else float("nan"),
        mcc=float(matthews_corrcoef(y, yhat)) if both else float("nan"),
        f1=float(f1_score(y, yhat, zero_division=0)),
        brier=float(brier_score_loss(y, p)),
        calibration_slope=slope, calibration_intercept=intercept,
        threshold=float(threshold))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "auroc",
                 n_boot: int = 2000, seed: int = 20260722
                 ) -> tuple[float, float]:
    """Patient-level bootstrap confidence interval for one metric."""
    rng = np.random.default_rng(seed)
    fn = {"auroc": roc_auc_score, "auprc": average_precision_score}[metric]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                           n_boot: int = 2000, seed: int = 20260722
                           ) -> dict[str, float]:
    """Paired bootstrap difference between two models on the same patients."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) -
                      roc_auc_score(y[idx], p2[idx]))
    if not deltas:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_value=float("nan"))
    d = np.asarray(deltas)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=float(obs), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_value=float(min(1.0, p)))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Youden-index threshold. Must be given out-of-fold predictions, never resubstitution ones."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])

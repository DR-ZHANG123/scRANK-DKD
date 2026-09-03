"""Comparator models: conventional expression models and rank-based models.

All feature selection, hyperparameter choice and thresholds are confined to the current
outer training set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from statsmodels.stats.multitest import multipletests

SEED = 20260722


def select_degs(X: np.ndarray, y: np.ndarray, genes: list[str],
                top_n: int = 50) -> tuple[np.ndarray, pd.DataFrame]:
    """Select the top differentially expressed genes within the training set."""
    t, p = ttest_ind(X[y == 1], X[y == 0], axis=0, equal_var=False)
    q = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    tab = pd.DataFrame(dict(gene=genes, t=t, pvalue=p, qvalue=q))
    order = np.argsort(-np.abs(np.nan_to_num(t)))[:top_n]
    return order, tab.iloc[order].reset_index(drop=True)


def fit_logistic(X, y, penalty="l2", C=1.0):
    m = LogisticRegression(penalty=penalty, C=C, solver="liblinear",
                           max_iter=5000, random_state=SEED)
    m.fit(X, y)
    return m


def fit_lasso_cv(X, y, groups=None):
    """L1-penalized logistic regression with the penalty chosen by inner cross-validation."""
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return fit_logistic(X, y, penalty="l1", C=0.1)
    m = LogisticRegressionCV(Cs=10, cv=n_splits, penalty="l1",
                             solver="liblinear", scoring="roc_auc",
                             max_iter=5000, random_state=SEED, n_jobs=1)
    m.fit(X, y)
    return m


def fit_rf(X, y):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                               max_features="sqrt", random_state=SEED,
                               n_jobs=4, class_weight="balanced")
    m.fit(X, y)
    return m


def fit_xgb(X, y):
    from xgboost import XGBClassifier
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, scale_pos_weight=neg / pos,
                      eval_metric="logloss", random_state=SEED,
                      n_jobs=4, tree_method="hist")
    m.fit(X, y)
    return m


def module_scores(rank: pd.DataFrame, programs: dict[str, list[str]]
                  ) -> np.ndarray:
    """Mean within-sample rank of each program's genes."""
    cols = []
    for genes in programs.values():
        present = [g for g in genes if g in rank.index]
        cols.append(rank.loc[present].mean(axis=0).values if present
                    else np.zeros(rank.shape[1]))
    return np.vstack(cols).T


def genomewide_pair_features(rank: pd.DataFrame, gene_pairs: list[tuple[str, str]]
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Build genome-wide rank-pair features from the top training-set differentially expressed genes."""
    a = rank.reindex([g for g, _ in gene_pairs]).values
    b = rank.reindex([g for _, g in gene_pairs]).values
    diff = a - b
    return (diff > 0).astype(np.int8), np.isfinite(diff)

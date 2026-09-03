"""Cohort loading and matrix assembly."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BULK = ROOT / "data_processed" / "bulk"
PAIR = ROOT / "data_processed" / "pair_matrix"

LOCO_COHORTS = {
    "GLOM": ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1"],
    "TUB": ["TUB_GSE30529", "TUB_ERCB1", "TUB_ERCB2"],
}
ALL_COHORTS = {
    "GLOM": ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1",
             "GLOM_ERCB2", "GLOM_GSE1009"],
    "TUB": ["TUB_GSE30529", "TUB_ERCB1", "TUB_ERCB2"],
}


@dataclass
class CohortData:
    """One cohort: expression, within-sample ranks, sample labels and metadata."""
    cohort: str
    samples: np.ndarray
    y: np.ndarray               # 1 = DKD, 0 = Control
    P: np.ndarray               # pairs x samples, int8
    M: np.ndarray               # pairs x samples, float32
    mask: np.ndarray            # pairs x samples, bool
    expr: pd.DataFrame
    rank: pd.DataFrame


@dataclass
class Compartment:
    name: str
    pairs: pd.DataFrame
    genes: np.ndarray
    idx_i: np.ndarray
    idx_j: np.ndarray
    category: np.ndarray
    cohorts: dict[str, CohortData]
    universe: list[str]


def _labels(cid: str) -> pd.Series:
    meta = pd.read_csv(BULK / f"{cid}_meta.tsv", sep="\t", index_col=0)
    return meta["group"]


def load_compartment(comp: str, cohorts: list[str] | None = None,
                     keep_groups: tuple[str, ...] = ("DKD", "Control")
                     ) -> Compartment:
    npz = np.load(PAIR / f"{comp}_pairs.npz", allow_pickle=True)
    pairs = pd.read_csv(PAIR / f"{comp}_pairs.tsv.gz", sep="\t")
    cohorts = cohorts or LOCO_COHORTS[comp]

    universe = None
    out: dict[str, CohortData] = {}
    for cid in cohorts:
        samples = np.asarray(npz[f"{cid}__samples"], dtype=object)
        groups = _labels(cid).reindex(samples)
        sel = groups.isin(keep_groups).values
        expr = pd.read_csv(BULK / f"{cid}_gene.tsv.gz", sep="\t", index_col=0)
        rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
        keep = samples[sel]
        out[cid] = CohortData(
            cohort=cid, samples=keep,
            y=(groups[sel] == "DKD").astype(int).values,
            P=npz[f"{cid}__P"][:, sel], M=npz[f"{cid}__M"][:, sel],
            mask=npz[f"{cid}__mask"][:, sel],
            expr=expr[list(keep)], rank=rank[list(keep)])
        universe = (set(expr.index) if universe is None
                    else universe & set(expr.index))

    return Compartment(name=comp, pairs=pairs,
                       genes=np.asarray(npz["gene_index"], dtype=object),
                       idx_i=npz["idx_i"], idx_j=npz["idx_j"],
                       category=npz["category"], cohorts=out,
                       universe=sorted(universe or []))


def stack(cds: list[CohortData], pair_idx: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack several cohorts into one matrix, keeping cohort identity for grouped splits."""
    sl = slice(None) if pair_idx is None else pair_idx
    P = np.hstack([c.P[sl] for c in cds])
    M = np.hstack([c.M[sl] for c in cds])
    mask = np.hstack([c.mask[sl] for c in cds])
    y = np.concatenate([c.y for c in cds])
    cid = np.concatenate([np.full(len(c.y), i) for i, c in enumerate(cds)])
    return P, M, mask, y, cid


def expression_matrix(cds: list[CohortData], genes: list[str],
                      standardize: bool = True) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray]:
    """Return the samples-by-genes matrix, labels and cohort identifiers for the given cohorts."""
    mats, ys, cids = [], [], []
    for i, c in enumerate(cds):
        sub = c.expr.reindex(genes)
        x = sub.values.T.astype(float)
        x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(x).any() else 0.0)
        if standardize:
            x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        mats.append(x)
        ys.append(c.y)
        cids.append(np.full(len(c.y), i))
    return np.vstack(mats), np.concatenate(ys), np.concatenate(cids)

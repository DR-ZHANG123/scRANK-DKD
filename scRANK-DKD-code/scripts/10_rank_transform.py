#!/usr/bin/env python
"""Within-sample percentile-rank encoding.

Ranking is performed over all genes of a cohort, not only the candidates, so that
platforms of different size map onto a common zero-to-one scale. This is what
makes the features independent of absolute expression and removes any need for
batch correction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
TAB = ROOT / "results" / "tables"

COMPARTMENTS = {
    "GLOM": ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1",
             "GLOM_ERCB2", "GLOM_GSE1009"],
    "TUB": ["TUB_GSE30529", "TUB_ERCB1", "TUB_ERCB2"],
}


def percentile_rank(expr: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank of every gene within each sample."""
    r = expr.rank(axis=0, method="average")
    g = expr.notna().sum(axis=0)
    return (r - 1).div(g - 1, axis=1)


def main() -> None:
    qc = []
    universe: dict[str, set[str]] = {}
    for comp, cohorts in COMPARTMENTS.items():
        gene_sets = []
        for cid in cohorts:
            expr = pd.read_csv(BULK / f"{cid}_gene.tsv.gz", sep="\t", index_col=0)
            rank = percentile_rank(expr)
            rank.to_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t")
            gene_sets.append(set(expr.index))
            print(f"[{cid}] rank {rank.shape[0]} genes x {rank.shape[1]} samples "
                  f"(range {rank.values.min():.3f}-{rank.values.max():.3f})")

        loco = [c for c in cohorts if not c.endswith(("GSE1009", "ERCB2"))
                or c == "TUB_ERCB2"]
        loco_sets = [s for c, s in zip(cohorts, gene_sets) if c in loco]
        inter = set.intersection(*loco_sets)
        universe[comp] = inter
        print(f"[{comp}] LOCO cohorts {loco}: {len(inter)} shared genes")

        for cid in cohorts:
            rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
            sub = rank.loc[sorted(inter & set(rank.index))]
            cc = np.corrcoef(sub.T.values)
            off = cc[np.triu_indices_from(cc, k=1)]
            qc.append(dict(compartment=comp, cohort=cid,
                           n_shared_genes=sub.shape[0], n_samples=sub.shape[1],
                           within_cohort_rank_r_median=round(float(np.median(off)), 3),
                           within_cohort_rank_r_min=round(float(off.min()), 3)))

    pd.Series({k: sorted(v) for k, v in universe.items()}).to_json(
        BULK / "gene_universe.json")
    pd.DataFrame(qc).to_csv(TAB / "T10_rank_qc.tsv", sep="\t", index=False)
    print("\n" + pd.DataFrame(qc).to_string(index=False))


if __name__ == "__main__":
    main()

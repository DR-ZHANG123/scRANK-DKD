#!/usr/bin/env python
"""Construction of cell-state-constrained within-sample gene rank pairs.

The candidate gene pool is defined entirely by the single-cell programs and never
looks at a bulk label, so this step carries no label information and cannot leak
across cohorts. Statistical screening of the pairs happens inside each outer
training set, in 12_loco_experiment.py.

For every pair (i, j) we record:
    P_sij = 1[R_si > R_sj]        direction
    M_sij = |R_si - R_sj|         rank margin
and label the pair by origin:
    1  within one program
    2  between programs of the same cell type
    3  across cell types (may reflect composition change, so kept separate)

Writes data_processed/pair_matrix/<compartment>_pairs.npz and _pairs.tsv."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
PROG = ROOT / "data_processed" / "programs"
OUT = ROOT / "data_processed" / "pair_matrix"
TAB = ROOT / "results" / "tables"

COHORTS = {
    "GLOM": ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1",
             "GLOM_ERCB2", "GLOM_GSE1009"],
    "TUB": ["TUB_GSE30529", "TUB_ERCB1", "TUB_ERCB2"],
}

MAX_GENES_PER_PROGRAM = 30
MAX_CANDIDATE_GENES = 400
MIN_PROGRAM_GENES = 8


def load_programs() -> pd.DataFrame:
    df = pd.read_csv(PROG / "programs_raw.tsv", sep="\t")
    usage = pd.read_csv(TAB / "T09_program_usage.tsv", sep="\t")
    keep_nmf = (set(usage[usage.p_mwu <= 0.1001].program)
                if not usage.empty else set())
    mask = (df.route == "DE") | df.program.isin(keep_nmf)
    print(f"[programs] {len(df)} total, {int(mask.sum())} retained "
          f"(DE {int((df.route=='DE').sum())} + NMF {len(keep_nmf)})")
    return df[mask].reset_index(drop=True)


def build_candidates(programs: pd.DataFrame,
                     universe: set[str]) -> tuple[list[str], pd.DataFrame]:
    """Map genes to their primary program; a gene in several programs takes the highest-ranked one."""
    rows, seen = [], {}
    for _, p in programs.iterrows():
        genes = [g for g in str(p.genes).split(",") if g in universe]
        genes = genes[:MAX_GENES_PER_PROGRAM]
        if len(genes) < MIN_PROGRAM_GENES:
            continue
        for rank, g in enumerate(genes):
            rows.append(dict(gene=g, program=p.program, cell_type=p.cell_type,
                             route=p.route, direction=p.direction,
                             rank_in_program=rank))
    if not rows:
        raise RuntimeError("candidate gene pool is empty: no overlap between program genes and the bulk platform")
    tab = pd.DataFrame(rows)
    primary = (tab.sort_values(["gene", "rank_in_program"])
               .drop_duplicates("gene", keep="first")
               .set_index("gene"))

    if len(primary) > MAX_CANDIDATE_GENES:
        n_prog = tab.groupby("gene").program.nunique()
        score = n_prog.reindex(primary.index).fillna(1) * 1000 - primary.rank_in_program
        primary = primary.loc[score.sort_values(ascending=False)
                              .head(MAX_CANDIDATE_GENES).index]
    del seen
    return sorted(primary.index), primary.loc[sorted(primary.index)]


def pair_category(a: pd.Series, b: pd.Series) -> int:
    if a.program == b.program:
        return 1
    if a.cell_type == b.cell_type:
        return 2
    return 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universes = json.loads((BULK / "gene_universe.json").read_text())
    programs = load_programs()

    summary = []
    for comp, cohorts in COHORTS.items():
        universe = set(universes[comp])
        genes, ann = build_candidates(programs, universe)
        gi = {g: i for i, g in enumerate(genes)}
        print(f"\n[{comp}] {len(genes)} candidate genes from "
              f"{ann.program.nunique()} programs / {ann.cell_type.nunique()} cell types")

        idx_i, idx_j, cat = [], [], []
        for a in range(len(genes)):
            for b in range(a + 1, len(genes)):
                idx_i.append(a)
                idx_j.append(b)
                cat.append(pair_category(ann.iloc[a], ann.iloc[b]))
        idx_i = np.array(idx_i, dtype=np.int32)
        idx_j = np.array(idx_j, dtype=np.int32)
        cat = np.array(cat, dtype=np.int8)

        pairs = pd.DataFrame(dict(
            pair_id=np.arange(len(idx_i)),
            gene_a=[genes[i] for i in idx_i], gene_b=[genes[j] for j in idx_j],
            category=cat,
            program_a=ann.iloc[idx_i].program.values,
            program_b=ann.iloc[idx_j].program.values,
            cell_a=ann.iloc[idx_i].cell_type.values,
            cell_b=ann.iloc[idx_j].cell_type.values))
        pairs.to_csv(OUT / f"{comp}_pairs.tsv.gz", sep="\t", index=False)
        ann.to_csv(OUT / f"{comp}_candidate_genes.tsv", sep="\t")

        store: dict[str, np.ndarray] = {}
        for cid in cohorts:
            rank = pd.read_csv(BULK / f"{cid}_rank.tsv.gz", sep="\t", index_col=0)
            avail = [g for g in genes if g in rank.index]
            R = np.full((len(genes), rank.shape[1]), np.nan, dtype=np.float32)
            R[[gi[g] for g in avail]] = rank.loc[avail].values
            diff = R[idx_i] - R[idx_j]               # pairs x samples
            store[f"{cid}__P"] = (diff > 0).astype(np.int8)
            store[f"{cid}__M"] = np.abs(diff).astype(np.float32)
            store[f"{cid}__mask"] = np.isfinite(diff)
            store[f"{cid}__samples"] = np.array(rank.columns, dtype=object)
            cov = float(np.isfinite(diff).mean())
            summary.append(dict(compartment=comp, cohort=cid,
                                n_pairs=len(idx_i),
                                genes_available=len(avail),
                                pair_coverage=round(cov, 4)))
            print(f"  [{cid}] candidate genes available {len(avail)}/{len(genes)}, "
                  f"pair coverage {cov:.3f}")

        store["gene_index"] = np.array(genes, dtype=object)
        store["idx_i"], store["idx_j"], store["category"] = idx_i, idx_j, cat
        np.savez_compressed(OUT / f"{comp}_pairs.npz", **store)
        print(f"  [write] {OUT/f'{comp}_pairs.npz'}  ({len(idx_i)} pairs)")

    tab = pd.DataFrame(summary)
    tab.to_csv(TAB / "T11_pair_coverage.tsv", sep="\t", index=False)
    print("\n" + tab.to_string(index=False))


if __name__ == "__main__":
    main()

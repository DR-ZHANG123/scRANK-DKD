#!/usr/bin/env python
"""Model comparison, pair interpretation and program-level risk decomposition.

1. Paired bootstrap comparison of each model's AUROC against the primary model and
   the DRGpair baseline;
2. Summary of the core pairs selected stably across folds (cell of origin, program,
   direction, attention);
3. Attention weights used to decompose patient risk to the level of cell programs."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import metrics as MT                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
MODELS = ROOT / "models"
PAIR = ROOT / "data_processed" / "pair_matrix"

REFERENCE = "DRGpair_LASSO"
PRIMARY = "scPair_LASSO"


def model_comparisons(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comp, held), sub in pred.groupby(["compartment", "held_out"]):
        wide = sub.pivot_table(index="sample", columns="model", values="p")
        y = sub.drop_duplicates("sample").set_index("sample")["y"].reindex(wide.index)
        for base in (REFERENCE, PRIMARY):
            if base not in wide.columns:
                continue
            for model in wide.columns:
                if model == base:
                    continue
                d = MT.paired_bootstrap_delta(y.values, wide[model].values,
                                              wide[base].values)
                rows.append(dict(compartment=comp, held_out=held,
                                 model=model, reference=base, **d))
    return pd.DataFrame(rows)


def pooled_comparisons(pred: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstrap deltas on pooled held-out patients.
    
    Predictions are converted to within-fold percentile ranks before pooling, so that
    models on different probability scales cannot contaminate the comparison."""
    rows = []
    pred = pred.copy()
    pred["p"] = (pred.groupby(["compartment", "held_out", "model"])["p"]
                 .rank(pct=True))
    for comp, sub in pred.groupby("compartment"):
        wide = sub.pivot_table(index=["held_out", "sample"], columns="model",
                               values="p")
        y = (sub.drop_duplicates(["held_out", "sample"])
             .set_index(["held_out", "sample"])["y"].reindex(wide.index))
        for base in (REFERENCE, PRIMARY):
            if base not in wide.columns:
                continue
            for model in wide.columns:
                if model == base:
                    continue
                d = MT.paired_bootstrap_delta(y.values, wide[model].values,
                                              wide[base].values)
                lo, hi = MT.bootstrap_ci(y.values, wide[model].values)
                rows.append(dict(compartment=comp, model=model, reference=base,
                                 pooled_auroc=float(MT.evaluate(
                                     y.values, wide[model].values, 0.5)["auroc"]),
                                 auroc_ci_low=lo, auroc_ci_high=hi, **d))
    return pd.DataFrame(rows)


def core_pairs(sel: pd.DataFrame) -> pd.DataFrame:
    """Pairs reproduced across outer folds, with their direction and effect size."""
    rows = []
    for comp, sub in sel.groupby("compartment"):
        n_folds = sub.held_out.nunique()
        agg = (sub.groupby(["pair_id", "gene_a", "gene_b", "category"])
               .agg(n_folds_selected=("held_out", "nunique"),
                    mean_stability=("stability", "mean"),
                    mean_delta=("delta", "mean"),
                    mean_reversal_DKD=("reversal_DKD", "mean"),
                    mean_reversal_Control=("reversal_Control", "mean"),
                    mean_margin=("median_margin", "mean"))
               .reset_index())
        agg["compartment"] = comp
        agg["fold_consistency"] = agg.n_folds_selected / n_folds
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["compartment", "fold_consistency", "mean_stability"],
                           ascending=[True, False, False])


def attach_annotation(core: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for comp, sub in core.groupby("compartment"):
        pairs = pd.read_csv(PAIR / f"{comp}_pairs.tsv.gz", sep="\t")
        frames.append(sub.merge(
            pairs[["pair_id", "program_a", "program_b", "cell_a", "cell_b"]],
            on="pair_id", how="left"))
    return pd.concat(frames, ignore_index=True)


def program_decomposition(sel: pd.DataFrame) -> pd.DataFrame:
    """Split patient risk across cell programs using the attention weights."""
    rows = []
    for (comp, held), sub in sel.groupby(["compartment", "held_out"]):
        f = MODELS / f"attention_{comp}_{held}.npy"
        if not f.exists():
            continue
        alpha = np.load(f)
        pairs = pd.read_csv(PAIR / f"{comp}_pairs.tsv.gz", sep="\t")
        ann = pairs.set_index("pair_id").loc[sub.pair_id.values]
        if alpha.shape[1] != len(ann):
            continue
        w = alpha.mean(axis=0)
        for side in ("a", "b"):
            tmp = pd.DataFrame(dict(cell=ann[f"cell_{side}"].values,
                                    program=ann[f"program_{side}"].values,
                                    weight=w / 2))
            rows.append(tmp.assign(compartment=comp, held_out=held))
    if not rows:
        return pd.DataFrame()
    allw = pd.concat(rows, ignore_index=True)
    return (allw.groupby(["compartment", "held_out", "cell", "program"])
            .weight.sum().reset_index()
            .sort_values(["compartment", "held_out", "weight"],
                         ascending=[True, True, False]))


def patient_program_scores(sel: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-patient, per-program score matrix."""
    rows = []
    for (comp, held), sub in sel.groupby(["compartment", "held_out"]):
        f = MODELS / f"attention_{comp}_{held}.npy"
        if not f.exists():
            continue
        alpha = np.load(f)
        pairs = pd.read_csv(PAIR / f"{comp}_pairs.tsv.gz", sep="\t")
        ann = pairs.set_index("pair_id").loc[sub.pair_id.values]
        if alpha.shape[1] != len(ann):
            continue
        samples = pred[(pred.compartment == comp) & (pred.held_out == held) &
                       (pred.model == "scDRP_DKD")]
        if len(samples) != alpha.shape[0]:
            continue
        cells = pd.Index(sorted(set(ann.cell_a) | set(ann.cell_b)))
        mat = np.zeros((alpha.shape[0], len(cells)))
        for j, (ca, cb) in enumerate(zip(ann.cell_a, ann.cell_b)):
            mat[:, cells.get_loc(ca)] += alpha[:, j] / 2
            mat[:, cells.get_loc(cb)] += alpha[:, j] / 2
        df = pd.DataFrame(mat, columns=cells)
        df.insert(0, "y", samples.y.values)
        df.insert(0, "p", samples.p.values)
        df.insert(0, "sample", samples["sample"].values)
        df.insert(0, "held_out", held)
        df.insert(0, "compartment", comp)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    pred = pd.read_csv(MET / "loco_predictions.tsv.gz", sep="\t")
    sel = pd.read_csv(TAB / "T12_selected_pairs.tsv.gz", sep="\t")

    cmp_fold = model_comparisons(pred)
    cmp_fold.to_csv(TAB / "T14_model_comparison_per_fold.tsv", sep="\t", index=False)
    cmp_pool = pooled_comparisons(pred)
    cmp_pool.to_csv(TAB / "T15_model_comparison_pooled.tsv", sep="\t", index=False)
    print("===== model comparison on pooled held-out samples =====")
    print(cmp_pool[["compartment", "model", "reference", "pooled_auroc",
                    "delta", "lo", "hi", "p_value"]].to_string(index=False))

    core = attach_annotation(core_pairs(sel))
    core.to_csv(TAB / "T16_core_pairs.tsv.gz", sep="\t", index=False)
    for thr, label in ((1.0, "all folds"), (2 / 3, ">=2/3 of folds")):
        top = core[core.fold_consistency >= thr - 1e-9]
        print(f"\ncore pairs selected in {label}: {len(top)}")
        print(top.head(15)[["compartment", "gene_a", "gene_b", "category",
                            "mean_delta", "mean_stability", "cell_a",
                            "cell_b"]].to_string(index=False))

    genes = []
    for comp, sub in core.groupby("compartment"):
        long = pd.concat([
            sub[["gene_a", "cell_a", "program_a", "n_folds_selected",
                 "mean_delta"]].rename(columns={
                     "gene_a": "gene", "cell_a": "cell", "program_a": "program"}),
            sub[["gene_b", "cell_b", "program_b", "n_folds_selected",
                 "mean_delta"]].rename(columns={
                     "gene_b": "gene", "cell_b": "cell", "program_b": "program"})])
        g = (long.groupby(["gene", "cell", "program"])
             .agg(n_pairs=("gene", "size"),
                  max_folds=("n_folds_selected", "max"),
                  mean_abs_delta=("mean_delta", lambda v: float(np.abs(v).mean())))
             .reset_index().assign(compartment=comp))
        genes.append(g)
    gene_tab = pd.concat(genes, ignore_index=True).sort_values(
        ["compartment", "max_folds", "n_pairs"], ascending=[True, False, False])
    gene_tab.to_csv(TAB / "T16b_core_genes.tsv", sep="\t", index=False)
    print("\nmost reproducible candidate genes per compartment:")
    print(gene_tab.groupby("compartment").head(10)[
        ["compartment", "gene", "cell", "program", "n_pairs",
         "max_folds"]].to_string(index=False))

    dec = program_decomposition(sel)
    dec.to_csv(TAB / "T17_program_attention.tsv", sep="\t", index=False)
    if not dec.empty:
        print("\ncell programs with the highest attention share per compartment:")
        print(dec.groupby(["compartment", "cell"]).weight.mean()
              .sort_values(ascending=False).head(12).to_string())

    pat = patient_program_scores(sel, pred)
    if not pat.empty:
        pat.to_csv(TAB / "T18_patient_program_scores.tsv.gz", sep="\t", index=False)
        print(f"\npatient-level program composition table: {pat.shape}")


if __name__ == "__main__":
    main()

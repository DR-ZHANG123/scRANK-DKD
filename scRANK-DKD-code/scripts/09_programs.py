#!/usr/bin/env python
"""Cell-state program discovery: route A (DE programs) + route B (consensus NMF).

Route A - per cell type, the genes up- and down-regulated in DKD, required to agree in
direction in the second single-cell dataset, giving programs with a definite direction and a nominal cell of origin.

Route B - consensus NMF within each major cell type (k-means consensus over L2-normalised
spectra from multiple random restarts), K chosen by stability, giving unsupervised gene
programs; DKD and control program usage are then compared.

Writes program definitions and usage under data_processed/programs/, plus
results/tables/T08_programs.tsv and T09_program_usage.tsv.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
TAB = ROOT / "results" / "tables"
OUT = PROC / "programs"

SEED = 20260722
DISCOVERY, REPLICATION = "GSE131882", "GSE209781"

# Each single-cell dataset has only 3 vs 3 donors, and at that sample size DESeq2 finds
# almost nothing at FDR < 0.05 (at most 2 genes per cell type in GSE131882). Discovery
# therefore uses nominal significance plus an effect size, and the substantive stability requirement is cross-dataset direction agreement.
FDR_STRICT = 0.10       # prefer genes meeting this adjusted P
P_NOMINAL = 0.01        # fallback: nominal P value
MIN_LFC = 0.585         # |log2FC| >= 0.585, i.e. 1.5-fold
MAX_GENES_PER_PROGRAM = 50
MIN_GENES_PER_PROGRAM = 10

MIN_CELLS_FOR_NMF = 500
K_RANGE = range(4, 11)
N_RESTARTS = 10
SUBSAMPLE_FRAC = 0.6
N_HVG_NMF = 1500
TOP_GENES_NMF = 50

# Technical genes may not form a program on their own
TECH_PREFIX = ("MT-", "RPS", "RPL", "MRPS", "MRPL", "HB")
TECH_GENES = {"MALAT1", "NEAT1", "XIST"}


def is_tech(gene: str) -> bool:
    return gene.startswith(TECH_PREFIX) or gene in TECH_GENES


# --------------------------------------------------------------------------- #
# route A: DE programs
# --------------------------------------------------------------------------- #
def de_programs() -> pd.DataFrame:
    """The two single-cell datasets serve as each other's discovery and validation set, each contributing the cell types it covers well."""
    tabs = {ds: pd.read_csv(TAB / f"T06_pseudobulk_de_{ds}.tsv.gz", sep="\t")
            for ds in (DISCOVERY, REPLICATION)}
    idx = {ds: t.set_index(["cell_type", "gene"])["log2FoldChange"]
           for ds, t in tabs.items()}

    rows = []
    for ds, disc in tabs.items():
        other = REPLICATION if ds == DISCOVERY else DISCOVERY
        for ct, sub in disc.groupby("cell_type"):
            if ct in ("LowQuality", "Unassigned"):
                continue
            hit = ((sub.padj < FDR_STRICT) | (sub.pvalue < P_NOMINAL)) & \
                  (sub.log2FoldChange.abs() >= MIN_LFC)
            sig = sub[hit.fillna(False)]
            ct_in_other = ct in set(tabs[other].cell_type)
            for direction, sel in (("up", sig[sig.log2FoldChange > 0]),
                                   ("down", sig[sig.log2FoldChange < 0])):
                sel = sel[~sel.gene.map(is_tech)]
                if sel.empty:
                    continue
                lfc2 = sel.gene.map(lambda g, c=ct, o=other:
                                    idx[o].get((c, g), np.nan))
                concordant = (np.sign(lfc2) == np.sign(sel.log2FoldChange))
                if ct_in_other:
                    # cell type is measurable in the other dataset -> require the same direction
                    keep = sel[concordant.fillna(False).values]
                    replicated = True
                else:
                    keep = sel
                    replicated = False
                keep = keep.sort_values("pvalue")
                genes = list(dict.fromkeys(keep.gene))[:MAX_GENES_PER_PROGRAM]
                if len(genes) < MIN_GENES_PER_PROGRAM:
                    continue
                rows.append(dict(
                    program=f"{ds[3:]}_{ct}_DE_{direction}", cell_type=ct,
                    route="DE", direction=direction, source_dataset=ds,
                    n_genes=len(genes), n_sig_total=int(len(sel)),
                    cross_dataset_required=replicated,
                    frac_replicated=(float(concordant.mean(skipna=True))
                                     if lfc2.notna().any() else np.nan),
                    genes=",".join(genes)))
    out = pd.DataFrame(rows)
    # If both datasets yield a program for the same (cell type, direction), keep the one with more genes
    out = (out.sort_values(["cell_type", "direction", "n_genes"],
                           ascending=[True, True, False])
           .drop_duplicates(["cell_type", "direction"], keep="first")
           .reset_index(drop=True))
    return out


# --------------------------------------------------------------------------- #
# route B: consensus NMF
# --------------------------------------------------------------------------- #
def prep_nmf_matrix(adata: sc.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Within-cell-type HVG selection plus non-negative variance scaling."""
    sub = adata.copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=N_HVG_NMF, flavor="seurat")
    hv = sub.var_names[sub.var.highly_variable & ~sub.var_names.map(is_tech)]
    X = sub[:, hv].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X / (X.std(axis=0, keepdims=True) + 1e-9)     # no centring, so values stay non-negative
    return np.maximum(X, 0), np.asarray(hv)


def consensus_nmf(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """cNMF-style consensus: each restart uses random initialisation and a cell subsample so
    that restarts genuinely differ; k-means consensus over L2-normalised spectra, with silhouette as the stability measure."""
    rng = np.random.default_rng(SEED)
    n_sub = max(200, int(SUBSAMPLE_FRAC * X.shape[0]))
    spectra = []
    for r in range(N_RESTARTS):
        idx = rng.choice(X.shape[0], size=min(n_sub, X.shape[0]), replace=False)
        model = NMF(n_components=k, init="random", random_state=SEED + r,
                    max_iter=400, tol=1e-4)
        model.fit(X[idx])
        H = model.components_
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        spectra.append(H)
    S = np.vstack(spectra)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    stability = float(silhouette_score(S, km.labels_)) if k > 1 else 1.0
    cons = np.vstack([S[km.labels_ == i].mean(0) for i in range(k)])
    return cons, stability


def nmf_programs(adata: sc.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    prog_rows, usage_rows = [], []
    for ct, idx in adata.obs.groupby("cell_type", observed=True).indices.items():
        if ct in ("Unassigned", "LowQuality") or len(idx) < MIN_CELLS_FOR_NMF:
            continue
        sub = adata[idx].copy()
        X, genes = prep_nmf_matrix(sub)

        best = None
        for k in K_RANGE:
            cons, stab = consensus_nmf(X, k)
            if best is None or stab > best[1]:
                best = (k, stab, cons)
        k, stab, cons = best
        print(f"  [{ct}] n={len(idx)} chose K={k} (silhouette={stab:.3f})")

        # Fix the consensus spectra and solve only for each cell's usage
        from sklearn.decomposition import non_negative_factorization
        W, _, _ = non_negative_factorization(
            X.astype(np.float64), H=cons.astype(np.float64), n_components=k,
            init="custom", update_H=False, max_iter=400,
            W=np.full((X.shape[0], k), 0.1, dtype=np.float64))
        usage = W / (W.sum(axis=1, keepdims=True) + 1e-9)

        for j in range(k):
            order = np.argsort(cons[j])[::-1][:TOP_GENES_NMF]
            prog_rows.append(dict(
                program=f"{ct}_NMF{j+1}", cell_type=ct, route="NMF",
                direction="unsigned", n_genes=len(order),
                k_selected=k, stability=round(stab, 3),
                genes=",".join(genes[order])))
            df = pd.DataFrame(dict(sample=sub.obs["sample"].values,
                                   group=sub.obs["group"].values,
                                   usage=usage[:, j]))
            per_donor = df.groupby(["sample", "group"], observed=True)["usage"].mean()
            for (sample, group), val in per_donor.items():
                usage_rows.append(dict(program=f"{ct}_NMF{j+1}", cell_type=ct,
                                       sample=sample, group=group,
                                       mean_usage=float(val)))
    return pd.DataFrame(prog_rows), pd.DataFrame(usage_rows)


def usage_stats(usage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prog, sub in usage.groupby("program"):
        a = sub[sub.group == "DKD"].mean_usage.values
        b = sub[sub.group == "Control"].mean_usage.values
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(program=prog, cell_type=sub.cell_type.iloc[0],
                         mean_DKD=round(float(a.mean()), 4),
                         mean_Control=round(float(b.mean()), 4),
                         delta=round(float(a.mean() - b.mean()), 4),
                         p_mwu=float(p)))
    out = pd.DataFrame(rows)
    if not out.empty:
        from statsmodels.stats.multitest import multipletests
        out["fdr"] = multipletests(out.p_mwu, method="fdr_bh")[1]
    return out.sort_values("p_mwu")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===== route A: DE programs =====")
    de = de_programs()
    print(de[["program", "cell_type", "direction", "source_dataset", "n_genes",
              "n_sig_total", "cross_dataset_required",
              "frac_replicated"]].to_string(index=False))

    print("\n===== route B: consensus NMF =====")
    adata = sc.read_h5ad(PROC / "scrna" / f"{DISCOVERY}_annotated.h5ad")
    nmf, usage = nmf_programs(adata)

    programs = pd.concat([de, nmf], ignore_index=True)
    programs.to_csv(OUT / "programs_raw.tsv", sep="\t", index=False)
    programs.drop(columns=["genes"]).to_csv(TAB / "T08_programs.tsv",
                                            sep="\t", index=False)
    usage.to_csv(OUT / "nmf_usage_donor.tsv", sep="\t", index=False)

    stats = usage_stats(usage)
    stats.to_csv(TAB / "T09_program_usage.tsv", sep="\t", index=False)
    print("\nNMF program usage (DKD vs Control, donor level):")
    print(stats.to_string(index=False))
    print(f"\n[write] {OUT/'programs_raw.tsv'} - {len(programs)} candidate programs")


if __name__ == "__main__":
    main()

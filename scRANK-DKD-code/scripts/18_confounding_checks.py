#!/usr/bin/env python
"""Confounding checks: batch collinearity, control-tissue provenance, cell attribution purity.

Three questions bound what the conclusions may claim:

1. Whether the hybridisation batch of the ERCB cohorts (H series / batch field) is nearly
   collinear with diagnosis. Within-sample ranking absorbs platform scale but not batch, so
   if batch and diagnosis coincide, cross-cohort performance cannot be attributed to disease alone.
2. Whether control tissue was obtained differently between compartments (tumour nephrectomy
   versus living donor biopsy). If glomerular controls are all tumour nephrectomies while
   tubulointerstitial controls are mostly donor biopsies, part of the compartment difference comes from how the tissue was taken.
3. Attribution purity of the cell-state programs. Single-cell data carry ambient RNA, so a
   program labelled with one cell type may be dominated by markers of another. This study
   presents a definite cell of origin as a selling point, so it has to be quantified.

Writes T27/T28/T29.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
META = ROOT / "data_raw" / "metadata"
PROG = ROOT / "data_processed" / "programs"
PAIR = ROOT / "data_processed" / "pair_matrix"
TAB = ROOT / "results" / "tables"

LOCO = {"GLOM": ["GLOM_GSE30528", "GLOM_GSE96804", "GLOM_ERCB1"],
        "TUB": ["TUB_GSE30529", "TUB_ERCB1", "TUB_ERCB2"]}


def cramers_v(table: np.ndarray) -> float:
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def batch_confounding() -> pd.DataFrame:
    """Strength of association between batch label and diagnosis label."""
    sheet = pd.read_csv(META / "sample_sheet.tsv", sep="\t")

    def batch_of(row) -> str | None:
        raw = str(row.raw_characteristics)
        m = re.search(r"batch:\s*(\S+)", raw)
        if m:
            return m.group(1)
        m = re.search(r"\b(H\d+)-", str(row.title))       # ERCB H series
        return m.group(1) if m else None

    sheet["batch"] = sheet.apply(batch_of, axis=1)
    rows = []
    for cid, cohorts in LOCO.items():
        for c in cohorts:
            meta = pd.read_csv(BULK / f"{c}_meta.tsv", sep="\t", index_col=0)
            sub = sheet[sheet.gsm.isin(meta.index)].copy()
            sub["group"] = meta["group"].reindex(sub.gsm).values
            sub = sub[sub.group.isin(["DKD", "Control"])]
            if sub.batch.isna().all():
                rows.append(dict(compartment=cid, cohort=c, n=len(sub),
                                 n_batches=0, cramers_v=np.nan,
                                 fisher_p=np.nan,
                                 note="no batch field in the platform annotation"))
                continue
            tab = pd.crosstab(sub.batch, sub.group)
            p = np.nan
            if tab.shape == (2, 2):
                p = float(fisher_exact(tab.values)[1])
            rows.append(dict(
                compartment=cid, cohort=c, n=len(sub),
                n_batches=int(sub.batch.nunique()),
                cramers_v=round(cramers_v(tab.values), 3), fisher_p=p,
                note=("batch perfectly separable from diagnosis" if tab.shape[0] > 1 and
                      (tab > 0).sum(axis=1).max() == 1 else "batch overlaps diagnosis")))
    return pd.DataFrame(rows)


def control_provenance() -> pd.DataFrame:
    """Tissue provenance of the control samples."""
    sheet = pd.read_csv(META / "sample_sheet.tsv", sep="\t")
    pat = [("tumour nephrectomy", r"tumor nephrectom|unaffected portion"),
           ("living donor", r"living donor|[-_]ld\d|pretransplant"),
           ("labelled control only", r"\bcontrol\b")]
    rows = []
    for cid, cohorts in LOCO.items():
        for c in cohorts:
            meta = pd.read_csv(BULK / f"{c}_meta.tsv", sep="\t", index_col=0)
            ctl = meta[meta.group == "Control"]
            sub = sheet[sheet.gsm.isin(ctl.index)]
            blob = (sub.title.astype(str) + " | " + sub.source.astype(str) +
                    " | " + sub.raw_characteristics.astype(str)).str.lower()
            counts = {}
            for label, rx in pat:
                counts[label] = int(blob.str.contains(rx, regex=True).sum())
            rows.append(dict(compartment=cid, cohort=c, n_control=len(sub),
                             **counts))
    return pd.DataFrame(rows)


def program_purity() -> pd.DataFrame:
    """For each program, how many of its genes are canonical markers of another cell type."""
    markers = yaml.safe_load(
        (ROOT / "configs" / "kidney_markers.yaml").read_text())
    gene2type: dict[str, set[str]] = {}
    for ct, genes in markers.items():
        for g in genes:
            gene2type.setdefault(g, set()).add(ct)

    progs = pd.read_csv(PROG / "programs_raw.tsv", sep="\t")
    used = set()
    for comp in LOCO:
        used |= set(pd.read_csv(PAIR / f"{comp}_candidate_genes.tsv",
                                sep="\t")["program"])

    rows = []
    for r in progs.itertuples():
        genes = str(r.genes).split(",")
        own = sum(1 for g in genes if r.cell_type in gene2type.get(g, set()))
        foreign = {}
        for g in genes:
            for t in gene2type.get(g, set()):
                if t != r.cell_type:
                    foreign[t] = foreign.get(t, 0) + 1
        top = max(foreign.items(), key=lambda kv: kv[1]) if foreign else ("", 0)
        rows.append(dict(
            program=r.program, cell_type=r.cell_type, route=r.route,
            n_genes=len(genes), n_own_markers=own,
            n_foreign_markers=sum(foreign.values()),
            top_foreign_type=top[0], top_foreign_n=top[1],
            used_in_candidate_pool=r.program in used,
            attribution_flag=("suspect" if top[1] > own else "ok")))
    return pd.DataFrame(rows)


def expression_specificity() -> pd.DataFrame:
    """Whether the cell type expressing a program's genes most highly matches the type the program is named after.

    More direct than counting markers: if a program labelled ENDO has most of its genes
    maximal in PERI, the claim of a definite cell of origin does not hold.
    Ambient RNA contamination presents exactly this way.
    """
    import scanpy as sc
    progs = pd.read_csv(PROG / "programs_raw.tsv", sep="\t")
    used = set()
    for comp in LOCO:
        used |= set(pd.read_csv(PAIR / f"{comp}_candidate_genes.tsv",
                                sep="\t")["program"])

    means = {}
    for ds in ("GSE131882", "GSE209781"):
        ad = sc.read_h5ad(ROOT / "data_processed" / "scrna" /
                          f"{ds}_annotated.h5ad")
        expr = ad.raw.to_adata() if ad.raw is not None else ad
        keep = ~ad.obs.cell_type.isin(["LowQuality", "Unassigned"]).values
        df = pd.DataFrame(index=expr.var_names)
        for ct in sorted(pd.unique(ad.obs.cell_type[keep])):
            m = keep & (ad.obs.cell_type.values == ct)
            X = expr[m].X
            df[ct] = np.asarray(X.mean(axis=0)).ravel()
        means[ds] = df

    rows = []
    for r in progs.itertuples():
        ds = getattr(r, "source_dataset", None)
        ds = ds if isinstance(ds, str) and ds in means else "GSE131882"
        mat = means[ds]
        genes = [g for g in str(r.genes).split(",") if g in mat.index]
        if not genes:
            continue
        top = mat.loc[genes].idxmax(axis=1)
        frac = float((top == r.cell_type).mean())
        dominant = top.value_counts().idxmax()
        rows.append(dict(
            program=r.program, cell_type=r.cell_type, route=r.route,
            source_dataset=ds, n_genes_checked=len(genes),
            frac_genes_max_in_own_type=round(frac, 3),
            dominant_expressing_type=dominant,
            frac_genes_max_in_dominant=round(
                float((top == dominant).mean()), 3),
            used_in_candidate_pool=r.program in used,
            attribution_flag=("consistent" if dominant == r.cell_type
                              else "reassigned")))
    return pd.DataFrame(rows)


def main() -> None:
    batch = batch_confounding()
    batch.to_csv(TAB / "T27_batch_confounding.tsv", sep="\t", index=False)
    print("===== association between batch and diagnosis =====")
    print(batch.to_string(index=False))

    prov = control_provenance()
    prov.to_csv(TAB / "T28_control_provenance.tsv", sep="\t", index=False)
    print("\n===== control-sample tissue provenance =====")
    print(prov.to_string(index=False))

    pur = program_purity()
    pur.to_csv(TAB / "T29_program_purity.tsv", sep="\t", index=False)
    print("\n===== program attribution purity (programs entering the candidate pool) =====")
    sub = pur[pur.used_in_candidate_pool]
    print(sub.sort_values("top_foreign_n", ascending=False)
          .head(15)[["program", "cell_type", "n_own_markers",
                     "n_foreign_markers", "top_foreign_type",
                     "top_foreign_n", "attribution_flag"]].to_string(index=False))
    n_suspect = int((sub.attribution_flag == "suspect").sum())
    print(f"\nPrograms in the pool with doubtful marker attribution: {n_suspect} / {len(sub)}")

    spec = expression_specificity()
    spec.to_csv(TAB / "T30_program_expression_specificity.tsv", sep="\t",
                index=False)
    used = spec[spec.used_in_candidate_pool]
    print("\n===== cell type with highest expression of each program's genes =====")
    print(used.sort_values("frac_genes_max_in_own_type")
          [["program", "cell_type", "frac_genes_max_in_own_type",
            "dominant_expressing_type", "frac_genes_max_in_dominant",
            "attribution_flag"]].to_string(index=False))
    n_re = int((used.attribution_flag == "reassigned").sum())
    print(f"\nPrograms in the pool whose cell attribution disagrees with expression: {n_re} / {len(used)}")


if __name__ == "__main__":
    main()

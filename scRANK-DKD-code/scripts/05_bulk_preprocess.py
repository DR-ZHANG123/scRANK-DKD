#!/usr/bin/env python
"""Collapse probes to gene symbols and assemble the per-compartment matrices.

Probes mapping to several genes are discarded and the highest-variance probe is
kept per gene. Cohorts are never combined or batch-corrected: the premise under
test is that within-sample ranking absorbs platform differences on its own."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "bulk"
META = ROOT / "data_raw" / "metadata"
OUT = ROOT / "data_processed" / "bulk"

COHORTS: dict[str, dict] = {
    "GLOM_GSE30528": dict(gse="GSE30528", source="rma", gpl="GPL571",
                          compartment="glomerulus", role="loco"),
    "GLOM_GSE96804": dict(gse="GSE96804", source="GSE96804_series_matrix.txt.gz",
                          gpl="GPL17586", compartment="glomerulus", role="loco"),
    "GLOM_ERCB1": dict(gse="GSE99339", source="GSE99339-GPL19184_series_matrix.txt.gz",
                       gpl="GPL19184", compartment="glomerulus", role="loco"),
    "GLOM_ERCB2": dict(gse="GSE99339", source="GSE99339-GPL19109_series_matrix.txt.gz",
                       gpl="GPL19109", compartment="glomerulus", role="specificity"),
    "GLOM_GSE1009": dict(gse="GSE1009", source="GSE1009_series_matrix.txt.gz",
                         gpl="GPL8300", compartment="glomerulus", role="direction_only"),
    "TUB_GSE30529": dict(gse="GSE30529", source="rma", gpl="GPL571",
                         compartment="tubulointerstitium", role="loco"),
    "TUB_ERCB1": dict(gse="GSE104954", source="GSE104954-GPL24120_series_matrix.txt.gz",
                      gpl="GPL24120", compartment="tubulointerstitium", role="loco"),
    "TUB_ERCB2": dict(gse="GSE104954", source="GSE104954-GPL22945_series_matrix.txt.gz",
                      gpl="GPL22945", compartment="tubulointerstitium", role="loco"),
}

MIN_PER_GROUP = 5


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def entrez_to_symbol() -> dict[str, str]:
    path = META / "Homo_sapiens.gene_info.gz"
    out = {}
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            f = line.split("\t")
            if f[0] != "9606":
                continue
            sym = f[10] if f[10] != "-" else f[2]
            if sym != "-":
                out[f[1]] = sym
    return out


def read_platform_table(gpl: str) -> pd.DataFrame:
    """Read a GEO platform annotation table, resolving Entrez identifiers where needed."""
    with gzip.open(META / f"{gpl}_table.txt.gz", "rt", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("ID\t"))
    body = "\n".join(lines[start:])
    return pd.read_csv(pd.io.common.StringIO(body), sep="\t",
                       dtype=str, on_bad_lines="skip")


def probe_to_symbol(gpl: str, e2s: dict[str, str]) -> pd.Series:
    tab = read_platform_table(gpl)
    tab = tab.set_index("ID")

    if "Gene Symbol" in tab.columns:                       # GPL571 / GPL8300
        sym = tab["Gene Symbol"]
        sym = sym.where(~sym.fillna("").str.contains("///"))
    elif "gene_assignment" in tab.columns:                 # GPL17586 (HTA-2.0)
        def parse(v: str) -> str | None:
            if not isinstance(v, str) or v.strip() in ("", "---"):
                return None
            syms = set()
            for part in v.split("///"):
                bits = [b.strip() for b in part.split("//")]
                if len(bits) > 1 and bits[1] not in ("", "---"):
                    syms.add(bits[1])
            return syms.pop() if len(syms) == 1 else None
        sym = tab["gene_assignment"].map(parse)
    elif "ENTREZ_GENE_ID" in tab.columns:
        sym = tab["ENTREZ_GENE_ID"].map(lambda v: e2s.get(str(v).strip()))
    else:
        raise ValueError(f"{gpl}: unrecognised annotation columns {list(tab.columns)}")

    sym = sym.dropna().astype(str).str.strip()
    return sym[(sym != "") & (sym != "---")]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def read_series_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
        df = pd.read_csv(fh, sep="\t", index_col=0, comment="!",
                         na_values=["", "NA", "null"])
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [str(c).strip('"') for c in df.columns]
    return df.dropna(how="all").astype(float)


def load_expression(cid: str, cfg: dict) -> pd.DataFrame:
    if cfg["source"] == "rma":
        df = pd.read_csv(OUT / f"{cfg['gse']}_rma_probe.tsv.gz",
                         sep="\t", index_col=0)
    else:
        df = read_series_matrix(RAW / cfg["gse"] / cfg["source"])
    if np.nanmax(df.values) > 100:
        floor = 1.0
        df = np.log2(df.clip(lower=floor))
        print(f"  [{cid}] linear intensities -> log2")
    return df


def collapse_to_gene(expr: pd.DataFrame, sym: pd.Series) -> pd.DataFrame:
    common = expr.index.intersection(sym.index)
    e = expr.loc[common]
    g = sym.loc[common]
    var = e.var(axis=1)
    order = np.lexsort((-var.values, g.values))
    e, g = e.iloc[order], g.iloc[order]
    keep = ~g.duplicated(keep="first")
    out = e[keep.values]
    out.index = g[keep.values].values
    out.index.name = "gene"
    return out.sort_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv(META / "sample_sheet.tsv", sep="\t")
    e2s = entrez_to_symbol()
    print(f"[annot] Entrez to Symbol: {len(e2s)} entries")

    sym_cache: dict[str, pd.Series] = {}
    rows = []
    for cid, cfg in COHORTS.items():
        gpl = cfg["gpl"]
        if gpl not in sym_cache:
            sym_cache[gpl] = probe_to_symbol(gpl, e2s)
        sym = sym_cache[gpl]

        expr = load_expression(cid, cfg)
        meta = sheet[(sheet.gse == cfg["gse"]) &
                     (sheet.gpl == gpl)].set_index("gsm")
        meta = meta.loc[[s for s in expr.columns if s in meta.index]]
        expr = expr[meta.index]

        gene = collapse_to_gene(expr, sym)
        gene = gene.loc[gene.notna().all(axis=1)]

        meta = meta.assign(cohort=cid, compartment=cfg["compartment"],
                           role=cfg["role"])
        gene.to_csv(OUT / f"{cid}_gene.tsv.gz", sep="\t")
        meta.to_csv(OUT / f"{cid}_meta.tsv", sep="\t")

        counts = meta.group.value_counts().to_dict()
        n_dkd, n_ctl = counts.get("DKD", 0), counts.get("Control", 0)
        usable = (n_dkd >= MIN_PER_GROUP and n_ctl >= MIN_PER_GROUP
                  and cfg["role"] == "loco")
        rows.append(dict(cohort=cid, gse=cfg["gse"], gpl=gpl,
                         compartment=cfg["compartment"], role=cfg["role"],
                         n_total=len(meta), n_DKD=n_dkd, n_Control=n_ctl,
                         n_otherCKD=len(meta) - n_dkd - n_ctl,
                         n_genes=gene.shape[0],
                         in_loco=bool(usable)))
        print(f"[{cid}] {gene.shape[0]} genes x {len(meta)} samples "
              f"| DKD={n_dkd} Ctrl={n_ctl} other={len(meta)-n_dkd-n_ctl} "
              f"| LOCO={'yes' if usable else 'no'}")

    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "results" / "tables" / "T02_cohort_definition.tsv",
               sep="\t", index=False)
    print("\n" + tab.to_string(index=False))


if __name__ == "__main__":
    main()

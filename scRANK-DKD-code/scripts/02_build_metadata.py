#!/usr/bin/env python
"""Build the per-sample metadata table from GEO series headers.

Assigns compartment, diagnosis group and platform, and records the reason each
retrieved cohort is kept or excluded."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "bulk"
OUT = ROOT / "data_raw" / "metadata"

DIAG_PATTERNS: list[tuple[str, str]] = [
    (r"focal segmental glomerulosclerosis and minimal change", "FSGS_MCD"),
    (r"fsgs\s*&\s*mcd", "FSGS_MCD"),
    (r"diabetic (nephropathy|kidney disease)|\bdiabetes\b|\bdkd\b|[-_]dn\d", "DKD"),
    (r"living donor|healthy|tumor nephrectom|unaffected portion|\bcontrol\b|"
     r"normal kidney|nephrectomy|[-_]ld\d", "Control"),
    (r"iga nephropathy|\bigan\b|[-_]igan\d", "IgAN"),
    (r"focal (and )?segmental glomerulosclerosis|\bfsgs\b|[-_]fsgs\d", "FSGS"),
    (r"minimal change disease|\bmcd\b|[-_]mcd\d", "MCD"),
    (r"membranous (glomerulo)?nephr(opathy|itis)|\bmgn\b|[-_]mgn\d", "MGN"),
    (r"lupus nephritis|systemic lupus|\bsle\b|[-_]sle\d|[-_]ln\d", "LN"),
    (r"hypertensive (nephropathy|nephrosclerosis)|nephrosclerosis|\bhtn\b|[-_]htn\d", "HTN"),
    (r"thin (basement )?membrane disease|\btmd\b|[-_]tmd\d", "TMD"),
    (r"rapidly progressive glomerulonephritis|vasculitis|\brpgn\b|[-_]rpgn\d", "RPGN"),
    (r"tubulointerstitial nephritis|\btin\b", "TIN"),
    (r"chronic kidney disease", "CKD_other"),
]

COMPARTMENT_OVERRIDE = {"GSE1009": "glomerulus"}


def read_header(path: Path) -> dict[str, list[str]]:
    """Read a GEO series-matrix header into a flat key to values mapping."""
    hdr: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
            key, vals = parts[0][1:], parts[1:]
            n = 0
            base = key
            while key in hdr:
                n += 1
                key = f"{base}#{n}"
            hdr[key] = vals
    return hdr


def norm_group(text: str) -> str | None:
    t = text.lower().strip()
    for pat, val in DIAG_PATTERNS:
        if re.search(pat, t):
            return val
    return None


def norm_compartment(tissue_text: str, gse: str) -> str | None:
    """Map the free-text tissue field onto glomerulus or tubulointerstitium."""
    t = tissue_text.lower()
    if "tubul" in t or "interstiti" in t:
        return "tubulointerstitium"
    if "glomerul" in t:
        return "glomerulus"
    return COMPARTMENT_OVERRIDE.get(gse)


def parse_series(path: Path, gse: str) -> pd.DataFrame:
    hdr = read_header(path)
    gsm = hdr["Sample_geo_accession"]
    n = len(gsm)
    title = hdr.get("Sample_title", [""] * n)
    source = hdr.get("Sample_source_name_ch1", [""] * n)

    chars = [[] for _ in range(n)]
    for key, vals in hdr.items():
        if key.startswith("Sample_characteristics_ch1"):
            for i, v in enumerate(vals):
                if v:
                    chars[i].append(v)

    gpls = hdr.get("Sample_platform_id", ["NA"] * n)

    rows = []
    for i in range(n):
        kv = {}
        for c in chars[i]:
            if ":" in c:
                k, v = c.split(":", 1)
                kv[k.strip().lower()] = v.strip()

        group = None
        for field in ("disease state", "diagnosis", "disease", "group", "phenotype"):
            if field in kv:
                group = norm_group(kv[field])
                if group:
                    break
        if group is None:
            group = norm_group(title[i]) or norm_group(source[i])

        comp = norm_compartment(kv.get("tissue", "") or source[i], gse)

        indiv = kv.get("individual") or kv.get("patient") or kv.get("subject")
        rows.append({
            "gse": gse,
            "gpl": gpls[i],
            "gsm": gsm[i],
            "title": title[i],
            "source": source[i],
            "compartment": comp,
            "group": group or "UNRESOLVED",
            "individual": indiv,
            "sex": (kv.get("sex") or kv.get("gender") or "").lower() or None,
            "age": kv.get("age"),
            "raw_characteristics": " ; ".join(chars[i]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for gse_dir in sorted(RAW.iterdir()):
        if not gse_dir.is_dir():
            continue
        for mat in sorted(gse_dir.glob("*series_matrix.txt.gz")):
            df = parse_series(mat, gse_dir.name)
            df.insert(2, "matrix_file", mat.name)
            frames.append(df)
            print(f"[{gse_dir.name}/{mat.name}] n={len(df)} "
                  f"groups={df['group'].value_counts().to_dict()}")
    sheet = pd.concat(frames, ignore_index=True)
    sheet.to_csv(OUT / "sample_sheet.tsv", sep="\t", index=False)
    print(f"\n[write] {OUT/'sample_sheet.tsv'}  ({len(sheet)} samples)")

    summary = (sheet.groupby(["gse", "gpl", "compartment", "group"])
               .size().rename("n").reset_index())
    summary.to_csv(ROOT / "results" / "tables" / "T01_cohort_overview.tsv",
                   sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

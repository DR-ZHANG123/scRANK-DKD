# scRANK-DKD

Analysis code for the manuscript *A leakage-controlled framework reveals the
importance of candidate-space provenance in cross-cohort kidney transcriptomics*.

`scRANK-DKD` is the analytical framework; `scPair-LASSO` is its primary penalized
rank-pair classifier.

**Code only.** All input data are public and are downloaded by the first script.
Everything the pipeline produces — `data_raw/`, `data_processed/`, `results/`,
`figures/` — is created when you run it and is not distributed here.

## Run

```bash
conda env create -f environment.yml      # Python 3.10: dkd_v3
conda env create -f environment_r.yml    # R 4.3, CEL parsing only: dkd_r
conda activate dkd_v3
bash run_all.sh
```

`run_all.sh` goes from download to final figures and stops at the first failure.
Budget several hours and about 10 GB of disk. All random seeds are fixed at
20260722.

## Layout

```
run_all.sh                       one-command reproduction
scripts/                         pipeline, numbered in execution order
  scdrp/                         shared package: data, screening, models,
                                 metrics, baselines, figure style
configs/kidney_markers.yaml      canonical cell-type markers (read by 07 and 18)
environment.yml                  Python environment
environment_r.yml                R environment
```

## The three scripts behind the paper's main claim

The finding is about the pipeline, not about a gene list: **how a candidate
feature space was constructed can matter as much as which algorithm selects from
it.** The single-cell prior earned its place here mainly by defining a candidate
pool without reference to the outcome labels of the cohorts being modelled.

| Script | What it does |
|---|---|
| `scripts/12b_unmatched_comparator.py` | Rebuilds the genome-wide comparator three ways on identical folds, one component at a time |
| `scripts/12c_cascade_necessity.py` | Reciprocal test: removes the same screening cascade from the framework's own candidate pool |
| `scripts/26_figure_comparator.py` | Draws both; reads result tables only, refits nothing |

To be precise about the independence claim: the candidate pool is independent of
the **bulk** outcome labels being evaluated, not of disease labels in general —
the single-cell datasets used their own DKD and control labels.

## Worth knowing before reusing this

- The single-cell constraint improved transfer in glomeruli only, and that gain is
  specific to a comparator carrying the same screening cascade
  (`12b_unmatched_comparator.py`).
- Gene pairing did not beat a plain module score for classification, and a
  program-aware deep set did not beat the linear aggregator.
- Cell-of-origin attribution failed an expression-purity check for 18 of 27
  programs (`18_confounding_checks.py`). Do not build on the programs' cell-type
  labels without repeating that check.
- Deconvolution-based composition adjustment returned non-physical proportions and
  supports no claim; `22_state_composition.py` and `23_deconv_complete.py` are
  retained as exploratory and are commented out of `run_all.sh`.
- The screening cascade costs about 0.10 AUROC of external transfer on a candidate
  space that was itself selected against the outcome labels, and nothing on one
  those labels never touched (`12c_cascade_necessity.py`). Define candidates
  without looking at the outcome, or drop the cascade.

Pieces worth lifting for other diseases: `scdrp/screening.py` (the pair screening
cascade), `10_rank_transform.py` and `11_pair_generation.py` (within-sample rank
encoding and constrained pair construction), `12_loco_experiment.py` (the
leave-one-cohort-out harness, including `oof_predict()`, which sets the decision
threshold from out-of-fold training predictions rather than resubstitution).

## Data

Public, from the Gene Expression Omnibus, fetched by `scripts/01_download_data.py`.
Single-cell: GSE131882, GSE209781. Bulk: GSE30528, GSE30529, GSE96804, GSE99339,
GSE104954. GSE30122 and GSE1009 are retrieved but excluded — GSE30122 shares
patients with two retained cohorts, GSE1009 contains technical replicates.

## Citation

See `CITATION.cff`. Archived snapshot with a permanent DOI:

```
DOI: [to be inserted on deposit]
```

## License

MIT — see `LICENSE`.

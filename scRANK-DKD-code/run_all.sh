#!/usr/bin/env bash
# scRANK-DKD full pipeline. Stops at the first failure so that no stage runs on
# stale intermediates.
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/dkd_v3/bin/python}"
RS="${RS:-$HOME/miniconda3/envs/dkd_r/bin/Rscript}"
cd "$(dirname "$0")"
mkdir -p results/logs

run() { echo "==> $*"; "$@"; }

run "$PY" scripts/01_download_data.py
run "$PY" scripts/02_build_metadata.py
run "$RS" scripts/03_rma_from_cel.R
run "$PY" scripts/03b_rma_summarize.py
run "$PY" scripts/03c_validate_rma.py
run "$RS" scripts/04_convert_scrna_rds.R
run "$PY" scripts/05_bulk_preprocess.py
run "$PY" scripts/06_scrna_qc.py
run "$PY" scripts/07_cell_annotation.py
run "$PY" scripts/08_pseudobulk_de.py
run "$PY" scripts/09_programs.py
run "$PY" scripts/10_rank_transform.py
run "$PY" scripts/11_pair_generation.py
run "$PY" scripts/12_loco_experiment.py
run "$PY" scripts/12b_unmatched_comparator.py
run "$PY" scripts/12c_cascade_necessity.py
run "$PY" scripts/13_interpretability.py
run "$PY" scripts/14_specificity_clinical.py
run "$PY" scripts/15_robustness.py
run "$PY" scripts/15b_negative_controls.py
run "$PY" scripts/16_figures_part1.py
run "$PY" scripts/17_figures_part2.py
run "$PY" scripts/18_confounding_checks.py
run "$PY" scripts/19_figure_validity.py
run "$PY" scripts/20_directed_rank_graph.py
run "$PY" scripts/24_confounding_sensitivity.py
run "$PY" scripts/25_injury_ordering.py
run "$PY" scripts/21_module_ablation.py
# exploratory only (not used in manuscript claims — unreliable deconvolution):
# run "$PY" scripts/23_deconv_complete.py
# run "$PY" scripts/22_state_composition.py

echo "==> manuscript"
cd manuscript && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "done: manuscript/main.pdf"

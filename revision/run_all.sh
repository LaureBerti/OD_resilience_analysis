#!/bin/bash
# Re-runs every experiment of the paper (CPU only, resumable: a unit is skipped if its result file is complete),
# then regenerates every table, figure and Supporting Information file.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY revision/code/download_extra_datasets.py
for E in E1 E1b E2 E2b1000 E2b2000 E2b5000 E3 E5 X5 X6 X7; do
  echo "=== $E"; $PY revision/code/run_resilience.py exp=$E run.n_jobs=${JOBS:-8}
done
$PY revision/code/analyze_E1.py
$PY revision/code/analyze_rest.py
$PY revision/code/make_si_tables.py out_dir=revision/SI
$PY revision/code/plot_fig1.py out_dir=revision/figures
$PY revision/code/plot_fig2.py out_dir=revision/figures
# optional: implementation checks and idle-machine runtimes
# git clone https://github.com/Qian-Lily/ADOD external/ADOD && $PY revision/code/validate_reimplementations.py
# $PY revision/code/time_runtime.py

#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY analysis/code/download_extra_datasets.py
for E in E1 E1b E2 E2b1000 E2b2000 E2b5000 E3 E5 X5 X6 X7; do
  echo "=== $E"; $PY analysis/code/run_resilience.py exp=$E run.n_jobs=${JOBS:-8}
done
$PY analysis/code/oblique_variance.py
$PY analysis/code/analyze_E1.py
$PY analysis/code/analyze_rest.py
$PY analysis/code/paper_numbers.py
if [ -d external/ADOD ]; then $PY analysis/code/validate_reimplementations.py; fi
$PY analysis/code/make_si_tables.py
$PY analysis/code/plot_fig1.py
$PY analysis/code/plot_fig2.py

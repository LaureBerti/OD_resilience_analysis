# Outlier Detection Methods Are Not Equally Resilient to Sampling

Outlier detection is often run on a random sample of a dataset because the full dataset is too large
for the detector. This repository measures **resilience to sampling**: how much the records a detector
flags on a simple random sample (without replacement) agree with those it flags on the full dataset.
Both measures are label-free:

- **ρ (Dice resilience):** overlap of the top-K records of the sample under the sample fit and under the
  full-data fit (K = ⌊qn⌋, q = 10%); with ties at the cut-off, the expected overlap when tied records are
  chosen at random, independently for the two fits;
- **ρ̃ (rank resilience):** Spearman correlation of the two score vectors on the sample.

The code, raw results of every replicate, and scripts that produce every table and figure of the paper
are in **`revision/`**. `scripts/` holds two files of an earlier version that are still used:
`scripts/data_loader.py` loads the data, and `scripts/method_wrappers.py` is the earlier set of detector
implementations that `revision/code/validate_reimplementations.py` compares with the reference
implementations. Everything else from the earlier version (its scripts, results, tables and figures)
is in `archive/submitted_version/`; it is superseded and does not underlie the paper.

> **Status:** manuscript under review (2026).

---

## Classes of detectors (18 detectors)

| Class | What the score depends on | Detectors |
|-------|---------------------------|-----------|
| **I** | global location and scale statistics | 3σ, BoxPlot, MAD, χ², Mahalanobis, ROD |
| **II** | empirical CDFs (tail probabilities, histograms) | COPOD, ECOD, HBOS |
| **III** | ensembles grown on internal subsamples (ψ = 256) | Isolation Forest, Extended Isolation Forest |
| **IV** | the k-nearest-neighbour graph | LOF, KNN, ABOD, Feature Bagging, ADOD |
| **G** | a globally fitted model (not assigned to a class) | k-means scoring, one-class SVM |

Implementations (`revision/code/detectors_v2.py`): PyOD 0.9.9 (COPOD, ECOD, HBOS, ABOD,
Feature Bagging, ROD), scikit-learn (LOF, KNN with the PyOD definition, one-class SVM, Isolation Forest,
Mahalanobis), isotree
(Extended Isolation Forest), and a CPU port of the authors' ADOD code
(https://github.com/Qian-Lily/ADOD; exact nearest-neighbour search in place of the GPU index).
3σ, BoxPlot, MAD, χ² and k-means scoring are implemented from their definitions.

## Repository structure

```
revision/
├── code/
│   ├── detectors_v2.py              the 18 detectors (in-sample scores)
│   ├── datasets_v2.py               datasets, size tiers, synthetic data (E7, E8)
│   ├── run_resilience.py            one runner for every experiment:  exp=E1 | E1b | E2 | E2b1000 |
│   │                                E2b2000 | E2b5000 | E3 | E5 | X5 (paper E6) | X6 (E7) | X7 (E8)
│   ├── analyze_E1.py, analyze_rest.py   every table and number of the Results
│   ├── plot_fig1.py, plot_fig2.py        Figs 1 and 2
│   ├── make_si_tables.py                 S1–S3 Tables
│   ├── validate_reimplementations.py     ADOD port vs authors' code; earlier vs reference implementations
│   ├── eif_direction_check.py            score direction of the earlier EIF implementation
│   ├── time_runtime.py                   runtimes for E5 (run on an idle machine)
│   ├── oblique_variance.py               per-tree depth variance, axis-parallel vs oblique trees
│   ├── download_extra_datasets.py        13 extra datasets for E2 (pinned ODDS/ADBench copies)
│   └── inventory.py                      completeness check of the results tree
├── experiments/<exp>/raw/*.csv      one row per replicate (dataset × method × condition)
├── experiments/<exp>/tables/        derived tables (CSV and LaTeX fragments)
├── experiments/validation/          implementation checks
├── SI/                              S1–S3 Tables (CSV)
├── figures/                         Fig1.tif, Fig2.tif
└── run_all.sh                       re-runs everything (resumable)
scripts/                             data loader and earlier detector wrappers (see above)
archive/submitted_version/           earlier version of the pipeline (superseded)
data/                                datasets (downloaded, not versioned)
```

---

## Prerequisites

- **Python 3.10+** (tested on 3.13)
- macOS or Linux
- Git

---

## Installation

### Option A — Standard install (requires cmake for PyOD's numba backend)

```bash
brew install cmake            # macOS; on Linux use the system package manager
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for exact pins
```

### Option B — cmake-free install (macOS + Python 3.13)

Installs PyOD without numba (AutoEncoder disabled; all 18 benchmark detectors work):

```bash
python3.13 -m venv .venv
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` installs the scientific stack, then `pyod==0.9.9 --no-deps` with a
minimal numba stub, and replaces a few sklearn-incompatible PyOD wrappers with direct
sklearn calls. The ROD baseline requires a small numpy-2.0 patch (`np.asarray`),
applied automatically.

---

## Dataset Download

Datasets are from the [ADBench benchmark](https://github.com/Minqi824/ADBench)
(binary `.npz`, not stored in the repo):

```bash
python download_datasets.py          # the 16 benchmark datasets (~50 MB)
```

The 16 datasets, by size tier:

| Tier | Datasets | N |
|------|----------|---|
| T1 (N < 1K) | lympho, glass, vertebral, ionosphere, breastw | 148–683 |
| T2 (1K–10K) | letter, musk, thyroid, optdigits, satellite, pendigits, annthyroid | 1.6K–7.2K |
| T3 (10K–100K) | shuttle, smtp | 49K–95K |
| T4 (≥ 100K) | skin, covertype | 245K–286K |

Substitutions vs. the original plan (ADBench availability): `vertebral` for arrhythmia;
`smtp`/`skin`/full-scale `covertype` for the large-scale sets.
**E1, E1b and E4** use all 16; **E5** uses the three largest (covertype, skin, smtp) at full size;
the other experiments are listed in Table 3 of the paper. **E2** also uses 13 further datasets
(`python revision/code/download_extra_datasets.py`).

> `data/*.npz` are git-ignored. The per-replicate results in `revision/experiments/*/raw/` let you
> reproduce every table and figure without re-running the experiments.

---

## Reproducing the paper

All experiments run on a CPU (the paper used one laptop, Intel Core i9-9980HK, 64 GB, no GPU).

```bash
python download_datasets.py                                  # 16 core datasets into data/
.venv/bin/pip install hydra-core omegaconf statsmodels scikit-posthocs isotree
.venv/bin/pip install combo --no-deps                        # needed by PyOD's Feature Bagging
bash revision/run_all.sh                                     # all experiments, tables, figures, SI
```

`run_all.sh` is resumable: a (dataset, method) unit is skipped when its CSV already holds the expected
number of replicates. With the result files included in this repository, the analysis steps alone
(`analyze_E1.py`, `analyze_rest.py`, `make_si_tables.py`, `plot_fig*.py`) reproduce every table and
figure in minutes. Datasets with more than 5,000 records are represented in E1–E4 and E6–E8 by one fixed
random subset of 5,000 records; E5 uses the full datasets.

Which script produces what (paths under `revision/experiments/` unless stated):

| Paper item | Script | Output |
|------------|--------|--------|
| Table 6 (E1 by tier), Table 7 (E4), E1 tests (Friedman, Wilcoxon–Holm, mixed model) | `revision/code/analyze_E1.py` | `E1/tables/` |
| E1b, E2 rates, Table 8 (E2b), E3, Table 9 (E5), the values of Table 10 (E6), E7, E8 | `revision/code/analyze_rest.py` | `E1b/`, `E2/`, `E2b5000/`, `E2b_n_vs_p.csv`, `E3/`, `E5/`, `X5/`, `X6/`, `X7/` `tables/` |
| E6, oblique-split check | `revision/code/oblique_variance.py` | `E6b/` |
| Fig 1, Fig 2 | `revision/code/plot_fig1.py`, `plot_fig2.py` | `revision/figures/` (with `out_dir=revision/figures`, as in `run_all.sh`) |
| S1–S3 Tables (S2 computed from `E1/raw/`), Table R1 of the response letter | `revision/code/make_si_tables.py` | `revision/SI/` (with `out_dir=revision/SI`, as in `run_all.sh`), `validation/implementations_vs_reference.csv` |
| S1 Table data (ADOD port vs authors' code), Table R1 data | `revision/code/validate_reimplementations.py` | `validation/reimpl_validation.csv` |
| E5 runtimes | `revision/code/time_runtime.py` | `E5/tables/runtime_idle.csv` |

Tables 1–5 (notation, classes, experiments, datasets, methods) are written by hand.

To check the ADOD implementation against the authors' code (also run by `run_all.sh` if
`external/ADOD` exists):

```bash
git clone https://github.com/Qian-Lily/ADOD external/ADOD
.venv/bin/python revision/code/validate_reimplementations.py
```

---

## Data Availability

This repository is the data-availability record for the paper.

- **Derived data** (the per-replicate resilience results underlying every table and figure of the
  paper): `revision/experiments/*/raw/*.csv`, with derived tables in `revision/experiments/*/tables/`,
  regenerated deterministically (fixed seeds) by `revision/code/`.
- **Benchmark datasets** (16 core datasets, plus 13 used only in E2): existing public third-party data from
  [ADBench](https://github.com/Minqi824/ADBench) (`adbench/datasets/Classical/`);
  fetch with `python download_datasets.py`. Full-scale Covertype/Skin also at the
  [UCI ML Repository](https://archive.ics.uci.edu/).
- **Persistent identifier:** this repository is archived on Zenodo; the concept DOI
  **`10.5281/zenodo.20829792`** (https://doi.org/10.5281/zenodo.20829792) resolves to the latest
  version, and each version also has its own DOI.


>  The benchmark datasets are already public via ADBench/UCI.

---

## Citation

```bibtex
@unpublished{bertiequille2026resilience,
  title   = {Outlier Detection Methods Are Not Equally Resilient to Sampling},
  author  = {Berti-\'{E}quille, Laure and Loh, Ji Meng},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

MIT License. Contact: laure.berti@ird.fr

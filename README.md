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
are in **`revision/`**. The folders `scripts/`, `results/`, `tables/` and `figures/` hold an earlier
version of the pipeline, kept for reference: `scripts/data_loader.py` is still used to load the data,
and `scripts/method_wrappers.py` is the earlier set of detector implementations that
`revision/code/validate_reimplementations.py` compares with the reference implementations.

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

Implementations (`revision/code/detectors_v2.py`): PyOD 0.9.9 (COPOD, ECOD, HBOS, KNN, ABOD,
Feature Bagging, ROD), scikit-learn (LOF, one-class SVM, Isolation Forest, Mahalanobis), isotree
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
│   ├── download_extra_datasets.py        13 extra datasets for E2 (pinned ODDS/ADBench copies)
│   └── inventory.py                      completeness check of the results tree
├── experiments/<exp>/raw/*.csv      one row per replicate (dataset × method × condition)
├── experiments/<exp>/tables/        derived tables (CSV and LaTeX fragments)
├── experiments/validation/          implementation checks
├── SI/                              S1–S3 Tables (CSV)
├── figures/                         Fig1.tif, Fig2.tif
└── run_all.sh                       re-runs everything (resumable)
scripts/, results/, tables/, figures/   earlier version of the pipeline (see above)
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
| T1 (N < 1K) | vertebral, breastw, glass, lympho | 148–683 |
| T2 (1K–10K) | annthyroid, ionosphere, letter, musk, optdigits, pendigits, satellite, thyroid | 1.6K–7.2K |
| T3 (10K–100K) | shuttle, smtp | 49K–95K |
| T4 (≥ 100K) | skin, covertype | 245K–286K |

Substitutions vs. the original plan (ADBench availability): `vertebral` for arrhythmia;
`smtp`/`skin`/full-scale `covertype` for the large-scale sets.
**E1/E1b** use all 16; **E4** uses 13 (12 in the Part A rank test + Satellite in Part B);
**E5** uses the three largest (covertype, skin, smtp).

> `data/*.npz` are git-ignored. Pre-computed `results/*.csv` let you reproduce the
> tables/figures without re-running the experiments.

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

To check the ADOD implementation against the authors' code:

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
- **Persistent identifier:** this repository is archived on
  [Zenodo](https://doi.org/10.5281/zenodo.20829793) with DOI
  **`10.5281/zenodo.20829793`** — the citable, long-term record required by PLOS


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

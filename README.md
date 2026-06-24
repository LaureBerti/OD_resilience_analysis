# Outlier Detection Methods Are Not Equally Resilient to Sampling

Outlier detection is rarely run on a complete dataset — it is applied to *samples*
(survey responses, database subsets, materialised views, streaming windows). This
project defines and measures **resilience to sampling**: how consistently a detector
flags the *same* records on a subsample as it would on the full dataset, under simple
random sampling without replacement (SRSWOR), stratified, and cluster designs.

Two complementary, label-free metrics are used: **Dice resilience** (ρ, set-overlap)
and **rank resilience** (ρ̃, Spearman correlation of score vectors). Seven experiments
(E1–E7) validate a four-class theory across **18 detectors** and **16 datasets**
(up to 286,048 records).

**Headline result:** at a 10% sampling fraction, local-density detectors (Class IV)
show **3.2× higher output divergence** (1−ρ̄ = 0.46 vs 0.14) than moment-based detectors
(Class I); Kruskal–Wallis H = 3142.7, p < 0.001 — confirming the theoretical class ordering.

> **Status:** manuscript under review (2026).

---

## The four resilience classes (18 detectors)

| Class | Statistical object | Bound | Detectors |
|-------|--------------------|-------|-----------|
| **I — Moment-based** | sample moments | CLT, O(1/√n) | 3Sigma, BoxPlot, MAD, ChiSquare |
| **II — Empirical-CDF** | empirical CDF | DKW (exponential) | COPOD, ECOD, HBOS |
| **III — Isolation** | internal subsample (ψ=256) | intrinsic, O(1/√T) | IForest, **EIF** |
| **IV — Local-density** | k-NN neighbourhood graph | hypergeometric, Ω(1/√(kp)) | LOF, KNN, OCSVM, Mahalanobis, KMeans, **ROD**, **ABOD**, **FeatureBagging**, **ADOD** |

The five newly added comparative baselines (bold) are:
**EIF** — Extended Isolation Forest (Hariri et al., IEEE TKDE 2021);
**ROD** — Rotation-based Outlier Detection (Almardeny et al., IEEE TKDE 2020);
**ABOD** — Angle-Based Outlier Detection (Kriegel et al., KDD 2008);
**Feature Bagging** (Lazarevic & Kumar, KDD 2005);
**ADOD** — Adaptive Density Outlier Detection (Qian et al., IEEE ICDM 2024).

> AutoEncoder is implemented but disabled when `torch` is unavailable; the benchmark
> uses the 18 detectors above.

---

## Repository Structure

```
OD_resilience/
├── scripts/
│   ├── data_loader.py         ADBench dataset loader (package or .npz fallback)
│   ├── method_wrappers.py     18 detector wrappers (common fit / decision_function interface)
│   ├── resilience_metrics.py  Dice (rho) and rank (rho_tilde) metrics
│   ├── exp_E1_class_partition.py   E1: class-partition hypothesis (Kruskal–Wallis)
│   ├── exp_E1b_multirate.py        E1b: multi-sampling-rate table (p = 1/5/10%)
│   ├── exp_E2_convergence.py       E2: convergence rate + IForest phase transition
│   ├── exp_E3_lof_kdep.py          E3: LOF/KNN k-dependence (Ω(1/√(kp)))
│   ├── exp_E4_rankbased.py         E4: rank-based (ρ̃) vs Dice (ρ) resilience
│   ├── exp_E5_largescale.py        E5: large-scale (3 largest datasets) + runtime
│   ├── exp_E6_stratified.py        E6: stratified vs SRSWOR
│   ├── exp_E7_cluster.py           E7: cluster sampling
│   ├── plot_E*.py / table_E*.py    Figures and LaTeX tables
│   ├── run_all.sh                  Run E1–E7 sequentially
│   └── run_exp_E{1-6}.sh           Per-experiment shell wrappers
├── results/                   Pre-computed CSVs (one per experiment, 18-detector run)
├── figures/                   Generated figures (PDF/PNG)
├── tables/                    Derived summary tables (CSV)
├── data/                      Dataset directory (populated by download_datasets.py)
├── download_datasets.py       Downloads ADBench .npz files
├── setup_env.sh               cmake-free install for Python 3.13 / macOS
├── requirements.txt           Loose version pins
├── requirements-lock.txt      Exact pinned versions
├── CITATION.cff               Machine-readable citation metadata
└── README.md                  This file
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

## Reproducing the Experiments

```bash
cd scripts/
bash run_all.sh              # E1–E7 sequentially
bash run_all.sh --skip-e5    # skip large-scale E5
bash run_all.sh --dry-run    # print plan, no execution
```

Individual experiments (write CSVs to `results/`, figures to `figures/`):

```bash
python exp_E1_class_partition.py   # E1: 18 methods × 16 datasets × 50 reps (Kruskal–Wallis)
python exp_E1b_multirate.py        # E1b: p in {1,5,10}% class means
python exp_E2_convergence.py       # E2: convergence exponents + IForest phase transition
python exp_E3_lof_kdep.py          # E3: LOF/KNN k-dependence
python exp_E4_rankbased.py         # E4: rho_tilde >= rho (15/18 methods), threshold sensitivity
python exp_E5_largescale.py        # E5: 3 largest datasets + runtime speedups
python exp_E6_stratified.py        # E6: stratified vs SRSWOR
python exp_E7_cluster.py           # E7: cluster sampling
```

Approximate total runtime ~8–24 h on a MacBook Pro (Apple M2, 16 GB); E1b and E5
dominate because the new EIF/ROD/ADOD detectors are compute-intensive at scale.

---

## Key Results

- **E1** — Class ordering confirmed: mean 1−ρ̄ at p=10% is 0.14 (I), 0.23 (II), 0.28 (III),
  0.46 (IV); Kruskal–Wallis H = 3142.7, p < 0.001.
- **E1b** — Ordering holds at every sampling fraction (p = 1/5/10%).
- **E4** — ρ̃ ≥ ρ for **15 of 18** methods (Wilcoxon, p < 0.05); the rank metric is
  threshold-free. Exceptions: LOF, ADOD (rank also degrades), BoxPlot (n.s.).
- **E5** — Ordering holds at scale (N up to 286K); IForest ρ ≈ 0.88 at p=1%;
  sampling gives ×1.7–×10.5 runtime speedups with negligible resilience loss.

---

## Data Availability

This repository is the data-availability record for the paper.
See **[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)** for the full statement, the
derived-data inventory, and per-dataset provenance.

- **Derived data** (the per-condition resilience results underlying every table/figure):
  `results/*.csv` and `tables/*.csv` — included here, regenerated deterministically
  (fixed seeds) by `scripts/`.
- **Benchmark datasets** (16): existing public third-party data from
  [ADBench](https://github.com/Minqi824/ADBench) (`adbench/datasets/Classical/`);
  fetch with `python download_datasets.py`. Full-scale Covertype/Skin also at the
  [UCI ML Repository](https://archive.ics.uci.edu/).
- **Persistent identifier:** archive a release on [Zenodo](https://zenodo.org) to mint a
  DOI (`10.5281/zenodo.XXXXXXX`) — required by PLOS; GitHub alone is not sufficient.
  Steps in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

> For a PLOS submission the derived data + code must be **publicly accessible at
> submission** (a placeholder DOI is accepted, finalised before publication) — not gated
> "upon acceptance". The benchmark datasets are already public via ADBench/UCI.

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

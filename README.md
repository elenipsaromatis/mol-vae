# mol-vae

A variational autoencoder (VAE) over SMILES strings, used as a latent-space
surrogate for Bayesian optimization (BO) of molecular LD50 toxicity. This is
the codebase for an MSc thesis project (Imperial College London).

## Overview

1. **Represent molecules.** A character-level VAE (`model.py`) is trained on
   SMILES strings to learn a continuous latent embedding of molecular
   structure, jointly supervised with a solubility regression head
   (`train_vae.py`). The training set is the overlap between the
   `Solubility_AqSolDB` and `LD50_Zhu` datasets (via
   [PyTDC](https://tdcommons.ai/)), so every molecule in the latent space
   also has a known LD50 label, even though LD50 itself is not a training
   target for the VAE.
2. **Search the latent space.** `bo/run_bo.py` fits a Gaussian Process (GP)
   on LD50 over that latent space and runs Bayesian optimization to find
   molecules with desirable (low) predicted toxicity. Several selection
   strategies are compared: UCB, Expected Improvement, Pareto-front
   selection, Pareto-Expert (an oracle-informed variant), and a Random
   baseline.
3. **Analyze the results.** Scripts under `analysis/` turn raw BO traces into
   convergence plots, top-N recovery tables, noise-robustness comparisons,
   and latent-space visualizations.

The BO objective is **pLD50** (`log(1/LD50)`, minimized) — see
[`analysis/TRUE_analysis/README.md`](analysis/TRUE_analysis/README.md) for
why, and for how that convention differs from earlier raw-LD50 experiments.

## Dependency: `pfgs-optimization`

Pareto-front selection and the BO dashboard schema depend on
**[`pfgs-optimization`](https://github.com/samstricker/pfgs-optimization)**
(Python package name `paretodo`), written by Sam Stricker. It is *not* on
PyPI — `requirements.txt` installs it as an editable local dependency:

```
-e ../pfgs-optimization
```

This means `pfgs-optimization` must be cloned as a **sibling directory** to
this repo before installing requirements:

```bash
cd ..
git clone https://github.com/samstricker/pfgs-optimization.git
cd mol-vae
```

so the two repos sit side by side:

```
parent-dir/
├── mol-vae/            (this repo)
└── pfgs-optimization/
```

`bo/run_bo.py` imports `paretodo.selection.selector.ParetoSelector` for
Pareto-front candidate selection, and `bo/dashboard_artifacts.py` writes
trace CSVs in the schema `paretodo`'s dashboard expects.

## Setup

```bash
# 1. Clone this repo and its sibling dependency (see above)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` also pulls in `PyTDC` (dataset access), `rdkit`
(cheminformatics), `torch`, `botorch`/`gpytorch` (GP/BO), `optuna`
(hyperparameter search), and `mlflow` (experiment tracking, logged locally
to `mlruns/`).

Raw property tables (`data/*.tab`) are cached locally from TDC on first run.

## Repo structure

```
mol-vae/
├── data.py                # SMILES loading, cleaning, tokenization, DataLoader construction
├── model.py                # VAE architecture (Encoder/Decoder, reparameterisation, property heads)
├── train_vae.py            # Trains the VAE (reconstruction + KL + solubility head) — produces checkpoints/
├── evaluate.py              # Reconstruction/generation evaluation utilities
├── hpo.py                  # Optuna hyperparameter search over VAE architecture/training config
├── run_hpo.sh               # PBS job wrapper for hpo.py
├── requirements.txt
│
├── data/                    # Cached raw property tables (AqSolDB, LD50_Zhu, CYP2C19/2D6)
├── checkpoints/              # Trained VAE checkpoints (*.pth, gitignored)
│
├── bo/                      # Bayesian optimization over the VAE latent space
│   ├── run_bo.py             # Main BO loop: UCB / Pareto / EI / Pareto-Expert / Random, multi-seed sweeps
│   ├── problem.py            # Loads a checkpoint, encodes the candidate pool, exposes LD50 labels to BO
│   ├── noise.py              # Synthetic measurement-noise model for the noise-robustness experiments
│   ├── dashboard_artifacts.py# Writes trace CSVs in the schema paretodo's dashboard reads
│   ├── repair_true_rank.py   # One-off repair tool for a historical ranking-direction bug in old traces
│   ├── compare_seed_performance.py # Legacy paired UCB-vs-Pareto seed comparison (raw-LD50 convention)
│   ├── config.toml            # Pareto-front sampler config (population size, generations, etc.)
│   ├── experiments/           # Standalone, throwaway feasibility scripts (not part of the reproducible sweep)
│   └── results_pLD50*/        # Raw BO run outputs (per seed/strategy bo_trace.csv), gitignored contents
│
├── analysis/                 # Post-hoc analysis of VAE + BO results
│   ├── analyze_vae.py          # VAE reconstruction/regression metrics, training diagnostics
│   ├── latent_space.py         # Latent-space PCA/3D visualizations colored by property
│   ├── sol_predictions.py      # Solubility prediction accuracy plots
│   ├── gp_predictions.py       # GP fit-quality metrics over the course of a BO run
│   ├── compare_bo_baselines.py # Multi-seed, multi-strategy BO comparison (convergence, top-N recovery)
│   ├── compare_noise_robustness.py # Compares BO performance across injected-noise levels
│   ├── analyze_bo.py           # Legacy raw-LD50 (maximize) BO analysis, kept for reference
│   ├── figures/, tables/       # Generated plots/tables (mostly gitignored)
│   └── TRUE_analysis/          # Canonical pLD50 (minimize) results — see its own README
│
├── hpc/                      # PBS cluster job scripts (VAE training, BO sweeps, per objective/strategy)
├── notebooks/                 # Exploratory Jupyter notebooks
├── tests/                     # pytest unit tests for data/model/evaluate/hpo
└── logs/                     # Raw stdout logs from cluster sweeps (untracked)
```

## Usage

```bash
# Train the VAE
python train_vae.py

# Run a BO sweep (see bo/config.toml and hpc/*.sh for the full flag set)
python bo/run_bo.py --checkpoint checkpoints/<name>_best.pth --objective minimize

# Compare BO strategies across seeds
python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/baseline_1000

# Run the test suite
pytest
```

Longer sweeps (multi-seed, 1000+ iterations) are run on a PBS cluster via
the scripts in `hpc/`, then pulled back locally for analysis.

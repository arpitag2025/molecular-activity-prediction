# Molecular Activity Prediction — Improved Benchmark

This repository explores prediction of molecular **pIC50** using linear regression and genetic-algorithm-inspired feature selection. The original implementation remains intact; the new `benchmark.py` provides a reproducible experimental comparison.

## Improvements

- **Genetic search bug fixed:** generations evolve from the previous population, with elitism, crossover, mutation, and deterministic random seeds.
- **Train-only fitness:** feature search uses training cross-validation, not validation/test targets.
- **Preprocessing in pipelines:** imputation and scaling are fitted on training folds.
- **Baselines:** mean prediction, all-feature linear regression, Ridge, Lasso, and univariate top-k selection.
- **Reporting:** validation/test R², RMSE, MAE; genetic convergence; selected descriptors; held-out predictions; reproducibility metadata.

## Install

```bash
python -m pip install -r requirements-benchmark.txt
```

## Run

Using the original six CSV files:

```bash
python benchmark.py --generations 15 --population 20 --features 8
```

Quick smoke test:

```bash
python benchmark.py --generations 2 --population 6 --features 5
```

**Recommended:** use the original, unscaled molecular data:

```bash
python benchmark.py --raw Alzheimer2.csv --target-unit nM
```

The original raw dataset is not committed here. Confirm the target concentration units before running.

## Output

Results are written to `results/`:

- `model_comparison.csv`: cross-validation, validation, and test metrics
- `genetic_history.csv`, `genetic_progress.png`: optimization convergence
- `selected_features.csv`: selected descriptor indices
- `test_predictions.csv`, `test_predictions.png`: observed and predicted values
- `metadata.json`: seed and dataset dimensions

## Scientific limitations

The original repository's six processed CSV files may have been scaled **before** the original train/test split. Combining and resplitting these files does not undo this historical leakage. For a defensible evaluation, use the raw data via `--raw` so all scaling happens within training pipelines.

This experiment uses random splits. Molecular scaffold splits, independent assays, and repeated-seed evaluation would provide stronger evidence of generalization. Baseline hyperparameters are fitted using internal CV; genetic candidates use training-only CV. The validation set supports model comparison, and test metrics are reported transparently, not used to guide genetic feature selection. Do not claim accuracy improvements until the experiments have been run and inspected.

## Attribution

Original repository and project belong to its author. The new benchmark and documentation were developed with AI assistance and should be reviewed and understood before being presented as independent implementation work.

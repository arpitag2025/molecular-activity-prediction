"""Leakage-aware, reproducible molecular pIC50 regression benchmark.

Run: python benchmark.py --generations 15 --population 20
Uses original six CSVs, recombined before a fresh split. For strict leakage-free
preprocessing, supply original unscaled data with --raw.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_pair(xfile, yfile):
    x = pd.read_csv(xfile, header=None).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(pd.read_csv(yfile, header=None).iloc[:, 0], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    return x.loc[valid].reset_index(drop=True).replace([np.inf, -np.inf], np.nan), y.loc[valid].to_numpy()


def load_data(raw=None, unit="nM"):
    if raw:
        data = pd.read_csv(raw, header=None)
        concentration = pd.to_numeric(data.iloc[:, 0], errors="coerce")
        if unit == "nM":
            y = 9 - np.log10(concentration.where(concentration > 0))
        elif unit == "M":
            y = -np.log10(concentration.where(concentration > 0))
        else:
            y = concentration
        valid = y.notna() & np.isfinite(y)
        x = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        return x.loc[valid].reset_index(drop=True).replace([np.inf, -np.inf], np.nan), y.loc[valid].to_numpy()
    pairs = [load_pair(a, b) for a, b in (
        ("Train-Data.csv", "Train-pIC50.csv"),
        ("Validation-Data.csv", "Validation-pIC50.csv"),
        ("Test-Data.csv", "Test-pIC50.csv"))]
    return pd.concat([p[0] for p in pairs], ignore_index=True), np.concatenate([p[1] for p in pairs])


def pipeline(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("model", model)])


def metrics(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


class GeneticFeatureSelector:
    """Optimize CV RMSE on training data; test targets are never inspected."""
    def __init__(self, n_features, k=8, population=20, generations=15, seed=42):
        self.n_features = n_features
        self.k = min(k, n_features)
        self.population = population
        self.generations = generations
        self.seed = seed

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        n, k = self.n_features, self.k
        cv = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        cache = {}
        def random_mask():
            m = np.zeros(n, dtype=bool)
            m[rng.choice(n, k, replace=False)] = True
            return m
        def repair(m):
            m = m.copy()
            on = np.flatnonzero(m)
            if len(on) > k:
                m[rng.choice(on, len(on) - k, replace=False)] = False
            elif len(on) < k:
                m[rng.choice(np.flatnonzero(~m), k - len(on), replace=False)] = True
            return m
        def fitness(m):
            key = tuple(np.flatnonzero(m))
            if key not in cache:
                cv_rmse = -cross_val_score(
                    pipeline(LinearRegression()), x.iloc[:, m], y, cv=cv,
                    scoring="neg_root_mean_squared_error").mean()
                cache[key] = float(cv_rmse)
            return cache[key]
        pop = [random_mask() for _ in range(self.population)]
        self.history_ = []
        self.best_score_ = float("inf")
        for generation in range(self.generations):
            ranked = sorted(((fitness(m), m) for m in pop), key=lambda t: t[0])
            if ranked[0][0] < self.best_score_:
                self.best_score_, self.support_ = ranked[0][0], ranked[0][1].copy()
            self.history_.append(self.best_score_)
            parents = [m for _, m in ranked[:max(2, self.population // 4)]]
            new_pop = [ranked[0][1].copy(), ranked[1][1].copy()]
            while len(new_pop) < self.population:
                i, j = rng.integers(len(parents), size=2)
                cut = int(rng.integers(1, n)) if n > 1 else 0
                child = np.r_[parents[i][:cut], parents[j][cut:]]
                child ^= rng.random(n) < 0.04
                new_pop.append(repair(child))
            pop = new_pop  # Critical fix: evolve from the latest generation.
        self.n_evaluated_ = len(cache)
        return self


def experiment(x, y, outdir="results", seed=42, generations=15, population=20, features=8):
    if len(y) < 25:
        raise ValueError("Need at least 25 samples")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=.25, random_state=seed)
    x_train, x_val, x_test = (frame.reset_index(drop=True) for frame in (x_train, x_val, x_test))
    k = min(features, x.shape[1])
    models = {
        "mean_baseline": pipeline(DummyRegressor()),
        "linear_all": pipeline(LinearRegression()),
        "ridge_all": pipeline(RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3)),
        "lasso_all": pipeline(LassoCV(cv=3, random_state=seed, max_iter=10000)),
        "univariate_top_k": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("select", SelectKBest(f_regression, k=k)),
            ("scaler", StandardScaler()), ("model", LinearRegression())]),
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    rows, predictions = [], {"actual_pIC50": y_test}
    for name, model in models.items():
        cv_rmse = float(-cross_val_score(model, x_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error").mean())
        model.fit(x_train, y_train)
        val_pred, test_pred = model.predict(x_val), model.predict(x_test)
        rows.append({"model": name, "train_cv_rmse": cv_rmse,
                     **{"validation_" + k: v for k, v in metrics(y_val, val_pred).items()},
                     **{"test_" + k: v for k, v in metrics(y_test, test_pred).items()}})
        predictions[name] = test_pred
    ga = GeneticFeatureSelector(x.shape[1], k, population, generations, seed).fit(x_train, y_train)
    selected = np.flatnonzero(ga.support_)
    model = pipeline(LinearRegression()).fit(x_train.iloc[:, selected], y_train)
    val_pred, test_pred = model.predict(x_val.iloc[:, selected]), model.predict(x_test.iloc[:, selected])
    rows.append({"model": "genetic_linear", "train_cv_rmse": ga.best_score_,
                 **{"validation_" + k: v for k, v in metrics(y_val, val_pred).items()},
                 **{"test_" + k: v for k, v in metrics(y_test, test_pred).items()}})
    predictions["genetic_linear"] = test_pred
    result = pd.DataFrame(rows).sort_values("validation_rmse")
    result.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"descriptor_index": selected}).to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame({"generation": np.arange(1, generations + 1),
                  "best_train_cv_rmse": ga.history_}).to_csv(out / "genetic_history.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "seed": seed, "n_train": len(y_train), "n_validation": len(y_val),
        "n_test": len(y_test), "n_features": x.shape[1],
        "genetic_candidates_evaluated": ga.n_evaluated_}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(range(1, generations + 1), ga.history_)
        ax.set(xlabel="Generation", ylabel="Best training CV RMSE", title="Genetic search convergence")
        fig.savefig(out / "genetic_progress.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.scatter(y_test, test_pred, alpha=.65)
        lo, hi = min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())
        ax.plot([lo, hi], [lo, hi], "--")
        ax.set(xlabel="Actual pIC50", ylabel="Predicted pIC50", title="Held-out test predictions")
        fig.savefig(out / "test_predictions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipping plots")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", help="Raw CSV with IC50/pIC50 in first column")
    p.add_argument("--target-unit", choices=["nM", "M", "pIC50"], default="nM")
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--generations", type=int, default=15)
    a = p.parse_args()
    x, y = load_data(a.raw, a.target_unit)
    print(experiment(x, y, a.outdir, a.seed, a.generations, a.population, a.features).to_string(index=False))


if __name__ == "__main__":
    main()

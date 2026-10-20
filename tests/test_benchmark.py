import numpy as np
import pandas as pd
from benchmark import GeneticFeatureSelector, experiment

def test_genetic_selector_deterministic():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(36, 10)))
    y = 2 * x.iloc[:, 2].to_numpy() + rng.normal(size=36) * .1
    a = GeneticFeatureSelector(10, 3, 6, 3, 4).fit(x, y)
    b = GeneticFeatureSelector(10, 3, 6, 3, 4).fit(x, y)
    assert np.array_equal(a.support_, b.support_)
    assert a.support_.sum() == 3
    assert len(a.history_) == 3

def test_end_to_end(tmp_path):
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(60, 10)))
    y = 3 * x.iloc[:, 0].to_numpy() + rng.normal(size=60) * .2
    table = experiment(x, y, tmp_path, generations=2, population=6, features=3)
    assert len(table) == 6
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "genetic_history.csv").exists()

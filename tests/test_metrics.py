import numpy as np

from fgpvdta.training.metrics import concordance_index, regression_metrics


def test_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert (
        m.mse == 0
        and np.isclose(m.ci, 1)
        and np.isclose(m.pearson, 1)
        and np.isclose(m.spearman, 1)
    )


def test_tie_half():
    assert np.isclose(concordance_index(np.array([1.0, 2.0]), np.array([3.0, 3.0])), 0.5)

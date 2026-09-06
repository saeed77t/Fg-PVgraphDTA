import numpy as np

from fgpvdta.data.deepdta import transform_davis_kd_nm


def test_transform():
    np.testing.assert_allclose(
        transform_davis_kd_nm(np.array([1.0, 10.0, 100.0])), np.array([9.0, 8.0, 7.0])
    )

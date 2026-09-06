import numpy as np
from PIL import Image

from fgpvdta.preprocessing.contact_images import (
    amino_acid_one_hot,
    build_historical_stacked_rgb,
    historical_collapsed_feature_map,
    save_historical_stacked_png,
    sinusoidal_encoding,
)


def test_green_exact():
    seq = "ACD"
    L = len(seq)
    pe = sinusoidal_encoding(L, 32)
    rm = np.repeat(pe.T[:, :, None], L, axis=2)
    cm = rm.transpose(0, 2, 1)
    oh = amino_acid_one_hot(seq)
    roh = np.repeat(oh[:, :, None], L, axis=2)
    coh = roh.transpose(0, 2, 1)
    expected = np.concatenate([rm, cm, roh, coh], axis=0).mean(axis=0)
    np.testing.assert_allclose(historical_collapsed_feature_map(seq), expected, atol=1e-7)


def test_blue_zero(tmp_path):
    seq = "ACD"
    c = np.array([[0, 0.2, 0.8], [0.2, 0, 0.4], [0.8, 0.4, 0]], dtype=np.float32)
    rgb = build_historical_stacked_rgb(c, seq)
    assert np.count_nonzero(rgb[..., 2]) == 0
    path = tmp_path / "x.png"
    save_historical_stacked_png(c, seq, path, write_metadata=False)
    arr = np.asarray(Image.open(path).convert("RGB"))
    assert arr[..., 2].max() == 0

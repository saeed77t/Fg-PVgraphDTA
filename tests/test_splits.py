import pandas as pd

from fgpvdta.data.splits import (
    assert_no_target_leakage,
    make_cold_target_split,
    make_random_split,
)


def frame():
    return pd.DataFrame(
        [
            {"Drug_ID": f"D{d}", "Target_ID": f"T{t}", "Y": float(d + t)}
            for t in range(10)
            for d in range(3)
        ]
    )


def test_random_deterministic():
    assert make_random_split(frame(), seed=42) == make_random_split(frame(), seed=42)


def test_cold_no_leak():
    f = frame()
    s = make_cold_target_split(f, seed=42)
    assert_no_target_leakage(f, s)

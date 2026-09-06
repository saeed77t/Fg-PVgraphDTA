"""Standalone worker compatible with the historical PconsC4 environment."""

import sys


def main():
    import numpy as np
    import pconsc4

    model = pconsc4.get_pconsc4()
    prediction = pconsc4.predict(model, sys.argv[1])
    matrix = np.asarray(prediction["cmap"])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Invalid PconsC4 contact map")
    np.save(sys.argv[2], matrix)


if __name__ == "__main__":
    main()

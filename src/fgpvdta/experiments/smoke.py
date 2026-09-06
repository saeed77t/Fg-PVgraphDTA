"""Small synthetic fixture exercising actual preprocessing and training code."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fgpvdta.data.schema import write_json
from fgpvdta.preprocessing.pipeline import prepare

from .presets import BASELINE, FG, PV
from .runner import RunOptions, predict_checkpoint, run_suite


def synthetic_inputs(root):
    root = Path(root)
    aln, contacts = root / "alignments", root / "contacts"
    aln.mkdir(parents=True, exist_ok=True)
    contacts.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, residue in enumerate("ACDEFGHI"):
        sequence = "AC" + residue + "GY"
        tid = f"T{target}"
        (aln / f"{tid}.aln").write_text(f"{sequence}\n{sequence}\n", encoding="utf-8")
        matrix = np.full((len(sequence), len(sequence)), 0.7, dtype=np.float32)
        np.fill_diagonal(matrix, 0)
        np.save(contacts / f"{tid}.npy", matrix)
        for drug, smiles in enumerate(("CCO", "CC(=O)O", "c1ccccc1N")):
            rows.append(
                {
                    "Drug_ID": f"D{drug}",
                    "Drug": smiles,
                    "Target_ID": tid,
                    "Target": sequence,
                    "Y": 5.0 + drug * 0.2 + target * 0.03,
                }
            )
    csv = root / "interactions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, aln, contacts


def run_smoke(output, include_pv=False):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be empty")
    torch.set_num_threads(2)
    csv, aln, contacts = synthetic_inputs(root / "inputs")
    data = prepare(csv, aln, contacts, root / "prepared")
    options = RunOptions(
        str(data),
        str(root / "runs"),
        seeds=(11, 22),
        epochs=1,
        batch_size=12,
        pv_batch_size=4,
        ae_epochs=1,
        ae_batch_size=2,
        test_fraction=0.25,
        validation_fraction=0.25,
        device="cpu",
        pretrained_resnet=False,
        synthetic=True,
    )
    results = run_suite("cold_target", options, [BASELINE, FG])
    prediction_path = root / "reloaded_predictions.csv"
    predict_checkpoint(results[0].parent, data, prediction_path)
    expected = pd.read_csv(results[0].parent / "test_predictions.csv")
    actual = pd.read_csv(prediction_path)
    np.testing.assert_allclose(expected.y_pred, actual.y_pred, atol=1e-6)
    if include_pv:
        pv = replace(PV, resnet="resnet18")
        results += run_suite("cold_target", replace(options, seeds=(11,)), [pv])
    write_json(
        root / "smoke_report.json",
        {
            "passed": True,
            "synthetic": True,
            "paper_results": False,
            "run_manifests": [str(p) for p in results],
            "reloaded_predictions_match": True,
            "included_pv": include_pv,
        },
    )
    print(f"Smoke test passed: {root / 'smoke_report.json'}")

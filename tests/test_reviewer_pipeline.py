"""Integration contracts for the final reviewer implementation."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from fgpvdta.analysis.aggregate import discover_observed_runs
from fgpvdta.analysis.reviewer import holm_adjust, paired_effects, paired_targets, summarize
from fgpvdta.data.schema import read_interactions
from fgpvdta.data.splits import DatasetSplit, validate_split_coverage
from fgpvdta.experiments.presets import BASELINE, FG, PV, SUITES, variants
from fgpvdta.experiments.runner import RunOptions, frozen_split, make_model, run_suite
from fgpvdta.experiments.smoke import synthetic_inputs
from fgpvdta.models.autoencoder import ContactMapImageDataset
from fgpvdta.preprocessing.feature_sets import (
    EXTRA_PATTERNS,
    ligand_features,
    protein_features,
    random_control,
)
from fgpvdta.preprocessing.functional_groups import atom_functional_group_matrix
from fgpvdta.preprocessing.pipeline import prepare, verify_prepared


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("prepared")
    csv, aln, contacts = synthetic_inputs(root / "inputs")
    return prepare(csv, aln, contacts, root / "prepared")


@pytest.mark.parametrize("count", [0, 10, 20, 30])
def test_feature_count_and_historical_prefix(count):
    mol = Chem.MolFromSmiles("CC(=O)O")
    values = ligand_features(mol, count)
    assert values.shape == (mol.GetNumAtoms(), count)
    np.testing.assert_array_equal(
        values[:, : min(count, 20)], atom_functional_group_matrix(mol)[:, : min(count, 20)]
    )
    assert protein_features("ACDX", count).shape == (4, count)


@pytest.mark.parametrize(
    "index,smiles",
    list(
        enumerate(
            [
                "CCCl",
                "C=C",
                "C#C",
                "CC=N",
                "CSSC",
                "CS(=O)(=O)N",
                "CS(=O)(=O)O",
                "OP(=O)(O)O",
                "CP(=O)(O)O",
                "c1ccncc1",
            ]
        )
    ),
)
def test_each_proposed_extra_smarts_matches_named_motif(index, smiles):
    assert Chem.MolFromSmiles(smiles).HasSubstructMatch(EXTRA_PATTERNS[index])


def test_random_control_is_stable_and_sparsity_matched():
    matrix = ligand_features(Chem.MolFromSmiles("CC(=O)O"), 20)
    a = random_control(matrix, "ligand:D1", 42)
    np.testing.assert_array_equal(a, random_control(matrix, "ligand:D1", 42))
    np.testing.assert_array_equal(a.sum(axis=1), matrix.sum(axis=1))
    assert not np.array_equal(a, matrix)


def test_pipeline_and_leading_zero_ids(tmp_path):
    csv, aln, contacts = synthetic_inputs(tmp_path / "inputs")
    frame = pd.read_csv(csv)
    frame.Drug_ID = frame.Drug_ID.map({"D0": "001", "D1": "002", "D2": "003"})
    frame.to_csv(csv, index=False)
    output = prepare(csv, aln, contacts, tmp_path / "prepared")
    frame, manifest = verify_prepared(output)
    assert set(frame.Drug_ID) == {"001", "002", "003"}
    assert (output / "ligands" / "k30" / "001_graph.pt").is_file()
    assert manifest["dataset_sha256"]
    (output / "ligands" / "k30" / "001_graph.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_prepared(output)


def test_duplicate_rows_and_invalid_split_fail(tmp_path):
    csv, _, _ = synthetic_inputs(tmp_path / "inputs")
    frame = pd.read_csv(csv)
    pd.concat([frame, frame.iloc[:1]]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_interactions(csv)
    split = DatasetSplit([0, 0], [], [1], 42, "random")
    with pytest.raises(AssertionError, match="Duplicate"):
        validate_split_coverage(pd.DataFrame({"x": [1, 2]}), split)


def test_cold_split_groups_identical_sequences_and_ae_train_subset(prepared, tmp_path):
    frame, _ = verify_prepared(prepared)
    frame.loc[frame.Target_ID == "T1", "Target"] = frame.loc[0, "Target"]
    options = RunOptions(str(prepared), test_fraction=0.25, validation_fraction=0.25)
    split, _, digest = frozen_split(frame, tmp_path, "cold_target", options)
    train = set(frame.iloc[split.train].Target)
    test = set(frame.iloc[split.test].Target)
    assert not train & test
    assert digest == frozen_split(frame, tmp_path, "cold_target", options)[2]
    ids = set(frame.iloc[split.train].Target_ID)
    ds = ContactMapImageDataset(prepared / "images" / "native", target_ids=ids)
    assert {p.name.removesuffix("_stacked.png") for p in ds.image_paths} == ids
    assert not ids & set(frame.iloc[split.test].Target_ID)


def test_all_graph_variant_dimensions():
    for suite in ("fg_ablation", "fg_count", "random_control"):
        for variant in variants(suite):
            model = make_model(variant, pretrained=False)
            ld = 78 + (variant.fg_count if variant.fg_variant in {"ligand_fg", "full"} else 0)
            pdim = 54 + (variant.fg_count if variant.fg_variant in {"protein_fg", "full"} else 0)
            edges = torch.tensor([[0, 1], [1, 0]])
            ligand = Batch.from_data_list([Data(x=torch.ones(2, ld), edge_index=edges)])
            protein = Batch.from_data_list([Data(x=torch.ones(2, pdim), edge_index=edges)])
            assert model(ligand, protein).shape == (1, 1)


def test_pv_loss_and_gate_gradients():
    model = make_model(replace(PV, resnet="resnet18"), dropout=0.0, pretrained=False)
    edges = torch.tensor([[0, 1], [1, 0]])
    ligand = Batch.from_data_list([Data(x=torch.ones(2, 78), edge_index=edges) for _ in range(2)])
    protein = Batch.from_data_list([Data(x=torch.ones(2, 54), edge_index=edges) for _ in range(2)])
    images = torch.randn(2, 3, 224, 224)
    latent = torch.randn(2, 128, requires_grad=True)
    result = model(ligand, protein, images, latent)
    (result.prediction.square().mean() + result.info_nce).backward()
    assert model.cnn_projection[0].weight.grad is not None
    assert latent.grad is None
    model.eval()
    with torch.no_grad():
        assert model(ligand, protein, images, latent).info_nce == 0


def test_observed_end_to_end_and_tables(prepared, tmp_path):
    options = RunOptions(
        str(prepared),
        str(tmp_path / "runs"),
        seeds=(11, 22),
        epochs=1,
        test_fraction=0.25,
        validation_fraction=0.25,
        device="cpu",
    )
    paths = run_suite("cold_target", options, [BASELINE, FG])
    runs = discover_observed_runs(tmp_path / "runs")
    assert len(runs) == 4
    summary = summarize(tmp_path / "runs", tmp_path / "tables", "dgraphdta", 100, [11, 22])
    assert set(summary.variant) == {"dgraphdta", "fggraphdta"}
    assert (tmp_path / "tables" / "paired_statistics.csv").is_file()
    assert run_suite("cold_target", options, [BASELINE, FG]) == paths
    with pytest.raises(ValueError, match="Changed run"):
        run_suite("cold_target", replace(options, learning_rate=0.002), [BASELINE])
    # These are synthetic fixtures: production smoke sets observed=false automatically.
    for path in paths:
        payload = json.loads(path.read_text())
        payload["observed"] = False
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="No observed"):
        discover_observed_runs(tmp_path / "runs")


def test_paired_units_and_holm():
    result = paired_effects([0, 0, 0], samples=100)
    assert result["p_two_sided"] == 1
    assert result["bootstrap_low"] == 0
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    with pytest.raises(ValueError, match="Unmatched seed"):
        paired_targets(pd.DataFrame({"seed": [1]}), pd.DataFrame({"seed": [2]}))


def test_all_suites_have_unique_named_variants():
    for suite in SUITES:
        items = variants(suite)
        assert len({v.name for v in items}) == len(items)

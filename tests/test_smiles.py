import pytest

from fgpvdta.preprocessing.smiles import standardize_smiles


def test_notebook_preserves_input():
    assert standardize_smiles("C(C)O", "notebook").standardized_smiles == "C(C)O"


def test_invalid_smiles_fails():
    with pytest.raises(ValueError):
        standardize_smiles("THIS_IS_NOT_SMILES")

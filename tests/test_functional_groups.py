from rdkit import Chem

from fgpvdta.preprocessing.functional_groups import (
    FG_NAMES,
    atom_functional_group_matrix,
    cleanup_group_names,
    notebook_residue_fg_vector,
)


def test_twenty():
    assert len(FG_NAMES) == 20


def test_cleanup():
    x = cleanup_group_names(["Phenol", "Alcohol", "Ester", "Ether"])
    assert "Alcohol" not in x and "Ether" not in x


def test_atom_matrix():
    m = Chem.MolFromSmiles("CC(=O)OCC")
    x = atom_functional_group_matrix(m, "notebook")
    assert x.shape == (m.GetNumAtoms(), 20)


def test_residue_dim():
    assert notebook_residue_fg_vector("Y").shape == (20,)

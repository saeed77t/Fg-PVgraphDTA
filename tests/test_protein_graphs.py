import numpy as np

from fgpvdta.preprocessing.protein_graphs import build_protein_graph


def test_dims(tmp_path):
    aln = tmp_path / "T.aln"
    aln.write_text("ACD\nACD\n")
    cm = tmp_path / "T.npy"
    np.save(cm, np.array([[0, 0.8, 0.1], [0.8, 0, 0.7], [0.1, 0.7, 0]], dtype=np.float32))
    assert build_protein_graph("T", "ACD", aln, cm, False).x.shape == (3, 54)
    assert build_protein_graph("T", "ACD", aln, cm, True).x.shape == (3, 74)

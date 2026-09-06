from fgpvdta.preprocessing.ligand_graphs import build_ligand_graph


def test_dims():
    assert build_ligand_graph("CCO", False).x.shape[1] == 78
    assert build_ligand_graph("CCO", True).x.shape[1] == 98


def test_bidirectional():
    g = build_ligand_graph("CC", False)
    edges = set(map(tuple, g.edge_index.T.tolist()))
    assert (0, 1) in edges and (1, 0) in edges

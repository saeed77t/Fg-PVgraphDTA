import numpy as np

from fgpvdta.preprocessing.contact_maps import contact_edge_index, load_contact_map


def test_threshold_bidirectional():
    m = np.array([[0, 0.8, 0.1], [0.8, 0, 0.6], [0.1, 0.6, 0]], dtype=np.float32)
    edges = set(map(tuple, contact_edge_index(m, "threshold", 0.5).T.tolist()))
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_zero_threshold_never_adds_self_edges():
    edges = set(map(tuple, contact_edge_index(np.zeros((3, 3)), threshold=0).T.tolist()))
    assert edges == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}


def test_raw_contact_diagonal_and_asymmetry_are_preserved(tmp_path):
    matrix = np.array([[0.8, 0.2], [0.6, 0.9]], dtype=np.float32)
    path = tmp_path / "contact.npy"
    np.save(path, matrix)
    np.testing.assert_array_equal(load_contact_map(path, 2, "none"), matrix)


def test_three_sparse_rr_records_are_not_a_dense_matrix(tmp_path):
    path = tmp_path / "contact.rr"
    path.write_text("1 2 0.8\n1 3 0.2\n2 3 0.6\n", encoding="utf-8")
    expected = np.array([[0, 0.8, 0.2], [0.8, 0, 0.6], [0.2, 0.6, 0]])
    np.testing.assert_allclose(load_contact_map(path, 3), expected)

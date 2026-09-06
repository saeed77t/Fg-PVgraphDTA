from fgpvdta.preprocessing.pssm import compute_residue_profile


def test_profile_shape(tmp_path):
    p = tmp_path / "T.aln"
    p.write_text("ACD\nACD\nAXD\n")
    assert compute_residue_profile(p, "ACD").shape == (3, 21)

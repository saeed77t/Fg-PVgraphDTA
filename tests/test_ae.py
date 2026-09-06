import torch

from fgpvdta.models.autoencoder import ContactMapAutoencoder, MaskedMSELoss


def test_mask():
    t = torch.tensor([[[[0.0, 0.05], [0.06, 1.0]]]])
    p = torch.zeros_like(t)
    expected = (0.06**2 + 1) / 2
    assert torch.isclose(MaskedMSELoss(0.05)(p, t), torch.tensor(expected), atol=1e-7)


def test_zero_blue_decoder():
    m = ContactMapAutoencoder(decoder_variant="rg_zero_blue")
    y = m.decode(torch.randn(1, 128))
    assert y.shape == (1, 3, 512, 512) and torch.count_nonzero(y[:, 2]) == 0

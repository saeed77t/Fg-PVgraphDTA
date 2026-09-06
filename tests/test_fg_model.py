import torch
from torch_geometric.data import Batch, Data

from fgpvdta.models.fggraphdta import FGGraphDTA


def g(d):
    return Data(x=torch.randn(4, d), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))


def test_forward():
    m = FGGraphDTA("full", dropout=0.0)
    out = m(Batch.from_data_list([g(98), g(98)]), Batch.from_data_list([g(74), g(74)]))
    assert out.shape == (2, 1)

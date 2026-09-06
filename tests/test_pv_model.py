import torch
from torch_geometric.data import Batch, Data

from fgpvdta.models.pvgraphdta import PVGraphDTA


def g(d):
    return Data(x=torch.randn(4, d), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))


def test_graph_baseline():
    m = PVGraphDTA(use_vision=False, use_ae_gating=False, use_infonce=False, dropout=0.0)
    out = m(Batch.from_data_list([g(78), g(78)]), Batch.from_data_list([g(54), g(54)]))
    assert out.prediction.shape == (2, 1)

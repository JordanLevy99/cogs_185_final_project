# https://github.com/jlevy44/WSI-GTFE/blob/master/notebooks/3_fit_gnn_model.ipynb
import torch, torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, DeepGraphInfomax, SAGEConv
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_geometric.nn import GINEConv
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F



class GCNNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_topology=[32,64,128,128], p=0.2, p2=0.0, drop_each=True):
        super(GCNNet, self).__init__()
        self.out_dim=out_dim
        self.convs = nn.ModuleList([GATConv(inp_dim, hidden_topology[0])]+[GATConv(hidden_topology[i],hidden_topology[i+1]) for i in range(len(hidden_topology[:-1]))])
        self.drop_edge = lambda edge_index: dropout_adj(edge_index,p=p2)[0]
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(hidden_topology[-1], out_dim)
        self.drop_each=drop_each

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        attention_weights=[]
        for conv in self.convs:
            if self.drop_each and self.training: edge_index=self.drop_edge(edge_index)
            x, attention = conv(x, edge_index, edge_attr,return_attention_weights=True)
            x = F.relu(x)
            attention_weights.append(attention)
        if self.training:
            x = self.dropout(x)
        x = self.fc(x)
        if return_attention: return x, attention_weights
        return x
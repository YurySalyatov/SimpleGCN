import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from utils import get_entropy, get_margin

class SimpleGCNModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=16, dropout=0.5):
        super(SimpleGCNModel, self).__init__()
        self.GCN1 = GCNConv(in_features, hidden_dim)
        self.GCN2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.Linear = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.GCN1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.GCN2(x, edge_index)

        return F.log_softmax(x, dim=1)


class SimpleGATModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=16, dropout=0.5, num_heads=8):
        super(SimpleGATModel, self).__init__()
        self.GAT1 = GATConv(in_features, hidden_dim, heads=num_heads)
        self.GAT2 = GATConv(num_heads * hidden_dim, hidden_dim)
        self.dropout = dropout
        self.Linear = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.GAT2(x, edge_index)

        x = self.Linear(x)

        self.margin.append(get_margin(x))
        self.entropy.append(get_entropy(x))

        return F.softmax(x, dim=1)

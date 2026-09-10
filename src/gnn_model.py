import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
import torch.nn.functional as F

class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Encoder for processing music structure graphs.
    """
    def __init__(self, in_channels=260, hidden_channels=[128, 64, 32], dropout=0.3):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        prev_channels = in_channels
        for channels in hidden_channels:
            self.convs.append(SAGEConv(prev_channels, channels))
            self.bns.append(nn.BatchNorm1d(channels))
            prev_channels = channels

    def forward(self, x, edge_index, batch=None):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        if batch is not None:
            return global_mean_pool(x, batch)
        return x

class GATEncoder(nn.Module):
    """
    Graph Attention Network (GAT) Encoder for processing music structure graphs.
    """
    def __init__(self, in_channels=260, hidden_channels=[128, 64], heads=2, dropout=0.3):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        prev_channels = in_channels
        for i, channels in enumerate(hidden_channels):
            concat = True if i < len(hidden_channels) - 1 else False
            self.convs.append(GATConv(prev_channels, channels, heads=heads, concat=concat, dropout=dropout))
            out_channels = channels * heads if concat else channels
            self.bns.append(nn.BatchNorm1d(out_channels))
            prev_channels = out_channels

    def forward(self, x, edge_index, batch=None):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        if batch is not None:
            return global_mean_pool(x, batch)
        return x

class GNNClassifier(nn.Module):
    """
    GNN Classifier with GraphSAGE or GAT encoder.
    """
    def __init__(self, encoder, graph_dim=32, num_classes=16):
        super(GNNClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_embedding = self.encoder(x, edge_index, batch)
        logits = self.classifier(graph_embedding)
        return logits, graph_embedding

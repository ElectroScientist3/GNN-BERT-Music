import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=256):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        # Project and L2-normalize
        x = F.normalize(x, p=2, dim=-1)
        return x

class ContrastiveDualEncoder(nn.Module):
    def __init__(self, gnn_encoder, bert_encoder, graph_dim=32, text_dim=768,
                 projection_dim=256, temperature=0.07):
        super(ContrastiveDualEncoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.bert_encoder = bert_encoder
        
        self.graph_proj = ProjectionHead(graph_dim, projection_dim)
        self.text_proj = ProjectionHead(text_dim, projection_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
    
    def forward(self, graph_data, input_ids, attention_mask):
        # Get graph embedding
        from torch_geometric.nn import global_mean_pool
        node_embeddings = self.gnn_encoder(graph_data.x, graph_data.edge_index)
        if hasattr(graph_data, 'batch') and graph_data.batch is not None:
            graph_emb = global_mean_pool(node_embeddings, graph_data.batch)
        else:
            graph_emb = node_embeddings.mean(dim=0, keepdim=True)
            
        # Project and normalize
        graph_proj = self.graph_proj(graph_emb)
        
        # Get BERT CLS
        text_emb = self.bert_encoder(input_ids, attention_mask=attention_mask)
        if isinstance(text_emb, tuple) or hasattr(text_emb, 'last_hidden_state'):
            if hasattr(text_emb, 'last_hidden_state'):
                text_emb = text_emb.last_hidden_state[:, 0, :]
            else:
                text_emb = text_emb[0][:, 0, :]
        elif text_emb.dim() == 3:
            text_emb = text_emb[:, 0, :]
            
        # Project and normalize
        text_proj = self.text_proj(text_emb)
        
        return graph_proj, text_proj
    
    def compute_loss(self, graph_emb, text_emb):
        # Compute similarity matrix: S = graph_emb @ text_emb.T / temperature
        temp = torch.clamp(self.temperature, min=1e-3, max=1.0)
        sim_matrix = torch.matmul(graph_emb, text_emb.T) / temp
        
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        loss_fn = nn.CrossEntropyLoss()
        # L_g2t = -mean(log(exp(S_ii) / sum_j(exp(S_ij))))
        g2t_loss = loss_fn(sim_matrix, labels)
        
        # L_t2g = -mean(log(exp(S_ii) / sum_j(exp(S_ji))))
        t2g_loss = loss_fn(sim_matrix.T, labels)
        
        total_loss = 0.5 * (g2t_loss + t2g_loss)
        return total_loss, g2t_loss, t2g_loss

def compute_retrieval_metrics(graph_emb, text_emb, ks=[1, 5, 10]):
    # Compute similarity matrix
    sim_matrix = torch.matmul(graph_emb, text_emb.T)
    
    num_samples = sim_matrix.size(0)
    labels = torch.arange(num_samples, device=sim_matrix.device).view(-1, 1)
    
    metrics = {}
    
    # For each graph, find top-K text matches (audio->caption retrieval)
    _, g2t_indices = torch.topk(sim_matrix, max(ks), dim=-1)
    for k in ks:
        correct = (g2t_indices[:, :k] == labels).sum().item()
        metrics[f'g2t_R@{k}'] = correct / num_samples
        
    # For each text, find top-K graph matches (caption->audio retrieval)
    _, t2g_indices = torch.topk(sim_matrix.T, max(ks), dim=-1)
    for k in ks:
        correct = (t2g_indices[:, :k] == labels).sum().item()
        metrics[f't2g_R@{k}'] = correct / num_samples
        
    return metrics

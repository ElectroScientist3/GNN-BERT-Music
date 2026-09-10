import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module mapping graph embeddings as queries and text hidden states as keys/values.
    """
    def __init__(self, graph_dim=32, text_dim=768, fused_dim=128, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.fused_dim = fused_dim
        
        # Project graph and text embeddings to a common dimension if needed, here we use text_dim for Q, K, V
        self.q_proj = nn.Linear(graph_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, text_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        
        # Output projection combining graph_emb and attention output
        self.out_proj = nn.Sequential(
            nn.Linear(graph_dim + text_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, graph_emb, text_hidden_states, text_attention_mask=None):
        """
        graph_emb: (batch, graph_dim)
        text_hidden_states: (batch, seq_len, text_dim)
        text_attention_mask: (batch, seq_len)
        """
        # MultiheadAttention expects Q: (batch, seq_len, embed_dim)
        q = self.q_proj(graph_emb).unsqueeze(1) # (batch, 1, text_dim)
        k = self.k_proj(text_hidden_states)     # (batch, seq_len, text_dim)
        v = self.v_proj(text_hidden_states)     # (batch, seq_len, text_dim)
        
        # Key padding mask: True for elements to ignore
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = (text_attention_mask == 0)
        
        attn_output, attn_weights = self.attention(q, k, v, key_padding_mask=key_padding_mask)
        # attn_output: (batch, 1, text_dim)
        attn_output = attn_output.squeeze(1) # (batch, text_dim)
        
        # Concat graph_emb and attn_output
        fused = torch.cat([graph_emb, attn_output], dim=1)
        fused_embedding = self.out_proj(fused)
        
        return fused_embedding, attn_weights

class EarlyConcatFusion(nn.Module):
    """
    Early concatenation fusion.
    """
    def __init__(self, graph_dim=32, text_dim=768, fused_dim=128):
        super(EarlyConcatFusion, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(graph_dim + text_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, graph_emb, text_cls):
        """
        graph_emb: (batch, graph_dim)
        text_cls: (batch, text_dim)
        """
        fused = torch.cat([graph_emb, text_cls], dim=1)
        fused_embedding = self.proj(fused)
        return fused_embedding

class GNNBERTFusionModel(nn.Module):
    """
    GNN-BERT Cross-Attention Multi-Task Fusion Model.
    """
    def __init__(self, gnn_encoder, bert_encoder, fusion_type='cross_attention',
                 graph_dim=32, num_tags=16, fused_dim=128, num_heads=4):
        super(GNNBERTFusionModel, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.bert_encoder = bert_encoder
        self.fusion_type = fusion_type
        
        text_dim = self.bert_encoder.bert.config.hidden_size
        
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(graph_dim=graph_dim, text_dim=text_dim, 
                                               fused_dim=fused_dim, num_heads=num_heads)
        elif fusion_type == 'early_concat':
            self.fusion = EarlyConcatFusion(graph_dim=graph_dim, text_dim=text_dim, fused_dim=fused_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # Multi-task heads
        self.tag_head = nn.Linear(fused_dim, num_tags)
        self.valence_head = nn.Linear(fused_dim, 1)
        self.arousal_head = nn.Linear(fused_dim, 1)

    def forward(self, graph_data, input_ids, attention_mask):
        # 1. GNN features
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        graph_embedding = self.gnn_encoder(x, edge_index, batch)
        
        # 2. BERT features
        text_hidden_states = self.bert_encoder.get_all_hidden_states(input_ids, attention_mask)
        
        # 3. Fusion
        attn_weights = None
        if self.fusion_type == 'cross_attention':
            fused_embedding, attn_weights = self.fusion(graph_embedding, text_hidden_states, attention_mask)
        else:
            text_cls = text_hidden_states[:, 0, :]
            fused_embedding = self.fusion(graph_embedding, text_cls)
            
        # 4. Multi-task prediction
        tag_logits = self.tag_head(fused_embedding)
        valence_pred = self.valence_head(fused_embedding)
        arousal_pred = self.arousal_head(fused_embedding)
        
        return tag_logits, valence_pred, arousal_pred, fused_embedding, attn_weights

import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    """Cross-modal cross-attention fusion module (core innovation of RCRank)"""
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # Cross attention: use SQL as Query, other modalities as Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1  # Prevent overfitting
        )
        # Feature fusion and dimensionality reduction (concatenate 4 modalities + attention output)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # 4 modalities + 1 attention output → 2*768
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),  # Reduce dimension to 768
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, sql_emb, plan_emb, ts_emb, log_emb):
        """
        Args:
            sql_emb: tensor(batch, 768), SQL features
            plan_emb: tensor(batch, 768), execution plan features
            ts_emb: tensor(batch, 768), time-series features
            log_emb: tensor(batch, 768), log features
        Returns:
            fused_emb: tensor(batch, 768), fused feature vector
        """
        # 1. Concatenate non-SQL modality features (as Key/Value)
        other_embs = torch.stack([plan_emb, ts_emb, log_emb], dim=1)  # (batch, 3, 768)
        # 2. Cross attention calculation (SQL-guided fusion of other modality features)
        sql_emb_unsqueeze = sql_emb.unsqueeze(1)  # (batch, 1, 768)
        attn_out, _ = self.cross_attn(
            query=sql_emb_unsqueeze,
            key=other_embs,
            value=other_embs
        )  # attn_out: (batch, 1, 768)
        # 3. Concatenate all features (4 modalities + attention output)
        concat_emb = torch.cat([
            sql_emb,
            plan_emb,
            ts_emb,
            log_emb,
            attn_out.squeeze(1)  # Remove extra dimension
        ], dim=1)  # (batch, 768*5)
        # 4. Feature fusion and dimensionality reduction
        return self.fusion_proj(concat_emb)
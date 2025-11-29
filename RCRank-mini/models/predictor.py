import torch
import torch.nn as nn

class RootCausePredictor(nn.Module):
    """Root cause ranking prediction head (outputs importance scores for 4 root cause categories)"""
    def __init__(self, hidden_dim=768, num_causes=4):
        super().__init__()
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(256, num_causes)  # Output scores for 4 root cause categories
        )
        
    def forward(self, fused_emb):
        """
        Args:
            fused_emb: tensor(batch, 768), multi-modal fused features
        Returns:
            pred_scores: tensor(batch, 4), importance scores for 4 root cause categories
        """
        return self.predict_head(fused_emb)
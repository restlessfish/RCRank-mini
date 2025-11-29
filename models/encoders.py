import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SQLBertEncoder(nn.Module):
    """SQL statement encoder (extract text features using BERT)"""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, sql_texts):
        """
        Args:
            sql_texts: list[str], list of SQL statements
        Returns:
            sql_emb: tensor(batch, 768), SQL feature vector
        """
        # BERT tokenization and encoding
        inputs = self.tokenizer(
            sql_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64  # Limit SQL text length
        )
        # Move input tensors to the device of BERT model (critical fix)
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        # Take [CLS] token feature as the overall SQL feature
        outputs = self.bert(** inputs)
        return outputs.last_hidden_state[:, 0, :]  # (batch, 768)

class PlanEncoder(nn.Module):
    """Execution plan encoder (process 3 statistical features)"""
    def __init__(self, input_dim=3, hidden_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)  # Normalization to improve training stability
        )
        
    def forward(self, plan_feat):
        """
        Args:
            plan_feat: tensor(batch, 3), 3 statistical features of execution plan
        Returns:
            plan_emb: tensor(batch, 768), execution plan feature vector
        """
        return self.fc(plan_feat)

class TimeSeriesEncoder(nn.Module):
    """Time series metrics encoder (process 5 statistical features)"""
    def __init__(self, input_dim=5, hidden_dim=768):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1  # Keep input and output length consistent
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling to 1 dimension
        
    def forward(self, ts_feat):
        """
        Args:
            ts_feat: tensor(batch, 5), 5 statistical features of time series metrics
        Returns:
            ts_emb: tensor(batch, 768), time series metrics feature vector
        """
        # Add channel dimension: (batch, 5) → (batch, 1, 5)
        ts_feat = ts_feat.unsqueeze(1)
        # Convolution to extract local features → pooling to compress dimension
        conv_out = self.conv(ts_feat)  # (batch, 768, 5)
        pool_out = self.pool(conv_out).squeeze(-1)  # (batch, 768)
        return pool_out

class LogEncoder(nn.Module):
    """Log feature encoder (process 3 statistical features)"""
    def __init__(self, input_dim=3, hidden_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh activation, suitable for feature compression
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, log_feat):
        """
        Args:
            log_feat: tensor(batch, 3), 3 statistical features of logs
        Returns:
            log_emb: tensor(batch, 768), log feature vector
        """
        return self.fc(log_feat)
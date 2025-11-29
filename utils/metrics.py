import torch
from scipy.stats import kendalltau
import pandas as pd
import json
import numpy as np

def kendall_tau_score(pred_scores, true_labels):
    """
    Calculate Kendall Tau coefficient (evaluate root cause ranking consistency)
    Args:
        pred_scores: list[tensor(4)], list of model-predicted root cause scores
        true_labels: list[tensor(4)], list of true root cause labels (One-Hot)
    Returns:
        avg_tau: float, average Kendall Tau coefficient (0-1, higher is better)
    """
    tau_scores = []
    for pred, true in zip(pred_scores, true_labels):
        # 1. Get root cause ranking from scores/labels (descending: higher scores first)
        pred_rank = torch.argsort(pred, descending=True).cpu().numpy()  # Predicted ranking
        true_rank = torch.argsort(true, descending=True).cpu().numpy()  # True ranking
        # 2. Calculate Kendall Tau (ignore p-value, take only coefficient)
        tau, _ = kendalltau(pred_rank, true_rank)
        # 3. Handle NaN (tau is NaN when all scores are the same, take 0)
        tau_scores.append(tau if not np.isnan(tau) else 0.0)
    # 4. Return average tau
    return sum(tau_scores) / len(tau_scores) if tau_scores else 0.0
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.encoders import SQLBertEncoder, PlanEncoder, TimeSeriesEncoder, LogEncoder
from models.fusion import CrossModalFusion
from models.predictor import RootCausePredictor
from utils.data_extractor import (
    extract_sql_text, extract_plan_features,
    extract_ts_features, extract_log_features, extract_root_cause
)
from utils.metrics import kendall_tau_score

# -------------------------- 1. Dataset Class (adapted to tpc_h.csv column names) --------------------------
class RawRCRankDataset(Dataset):
    def __init__(self, csv_path, sample_size=100):
        """
        Load tpc_h.csv data and automatically extract core features
        Core columns: query(SQL), plan_json(execution plan), log_all(logs), timeseries(time-series), multilabel(root cause labels)
        Skipped columns: index_x, opt_label_rate, duration, temp (unused)
        """
        # Keep only 5 core columns to avoid interference from useless fields
        core_cols = ["query", "plan_json", "log_all", "timeseries", "multilabel"]
        self.data = pd.read_csv(
            csv_path, 
            usecols=core_cols,  # Read only core columns
            encoding="utf-8"    # Avoid Chinese encoding errors
        ).head(sample_size)     # Take only first sample_size rows (50-100 rows)
        
        # Filter rows with empty values (ensure each row has at least SQL and root cause labels to avoid training errors)
        self.data = self.data.dropna(subset=["query", "multilabel"])
        
        # Print loading results (for debugging data volume)
        print(f"✅ tpc_h.csv data loaded successfully:")
        print(f"   - Original extracted: {sample_size} rows")
        print(f"   - After filtering empty values: {len(self.data)} valid rows")
        print(f"   - Core columns: {core_cols}")

    def __len__(self):
        """Return total number of dataset rows"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Extract single row by index, return format directly usable by model"""
        row = self.data.iloc[idx]
        return {
            "sql": extract_sql_text(row),          # SQL statement (string)
            "plan": extract_plan_features(row),    # Execution plan features (tensor(3,))
            "timeseries": extract_ts_features(row),# Time-series features (tensor(5,))
            "log": extract_log_features(row),      # Log features (tensor(3,))
            "label": extract_root_cause(row)       # Root cause labels (tensor(4,), One-Hot)
        }

# -------------------------- 2. Complete Model Assembly (reuse core logic) --------------------------
class MiniRCRank(nn.Module):
    def __init__(self, num_causes=4):
        super().__init__()
        # 1. Modality encoders (match feature dimensions)
        self.sql_encoder = SQLBertEncoder()                # SQL→tensor(768,)
        self.plan_encoder = PlanEncoder(input_dim=3, hidden_dim=768)  # 3→768
        self.ts_encoder = TimeSeriesEncoder(input_dim=5, hidden_dim=768)#5→768
        self.log_encoder = LogEncoder(input_dim=3, hidden_dim=768)    #3→768
        # 2. Cross-modal fusion (cross-attention, core of the paper)
        self.fusion = CrossModalFusion(hidden_dim=768)
        # 3. Root cause prediction head (output scores for 4 root cause categories)
        self.predictor = RootCausePredictor(hidden_dim=768, num_causes=num_causes)
        
    def forward(self, sql_texts, plan_feat, ts_feat, log_feat):
        """Forward propagation: input 4-modality features, output root cause scores"""
        # Individual modality encoding
        sql_emb = self.sql_encoder(sql_texts)
        plan_emb = self.plan_encoder(plan_feat)
        ts_emb = self.ts_encoder(ts_feat)
        log_emb = self.log_encoder(log_feat)
        # Cross-modal fusion
        fused_emb = self.fusion(sql_emb, plan_emb, ts_emb, log_emb)
        # Root cause prediction
        return self.predictor(fused_emb)

# -------------------------- 3. Main Training and Demonstration Logic --------------------------
if __name__ == "__main__":
    # -------------------------- Configuration Parameters (adjustable as needed) --------------------------
    CONFIG = {
        "csv_path": "data/tpc_h.csv",  # Key: path to tpc_h.csv (ensure it's in data folder)
        "sample_size": 4000,             # Training data volume 
        "batch_size": 4,               # Batch size (set to 2 if memory is low to avoid overflow)
        "epochs": 10,                  # Training epochs (10 epochs enough to verify core logic)
        "lr": 1e-4,                    # Learning rate (default 1e-4, no need to modify)
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto select device
    }

    # -------------------------- Step 1: Load Data --------------------------
    try:
        dataset = RawRCRankDataset(
            csv_path=CONFIG["csv_path"],
            sample_size=CONFIG["sample_size"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,        # Shuffle data to improve training效果
            drop_last=False,     # Don't drop last incomplete batch
            pin_memory=True      # Accelerate data transfer to GPU (if available)
        )
    except Exception as e:
        print(f"\n❌ Data loading failed! Error reason: {e}")
        print(f"   Please check: 1. Is tpc_h.csv in data/ folder? 2. Are column names query/plan_json/log_all/timeseries/multilabel?")
        exit()  # Exit if loading fails to avoid subsequent errors

    # -------------------------- Step 2: Initialize Model, Optimizer, Loss Function --------------------------
    model = MiniRCRank(num_causes=4).to(CONFIG["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=1e-5  # L2 regularization to prevent overfitting
    )
    criterion = nn.MSELoss()  # Mean squared error loss (suitable for root cause score regression)

    # -------------------------- Step 3: Training Loop --------------------------
    print(f"\n🚀 Start training (device: {CONFIG['device']}, batch size: {CONFIG['batch_size']}, epochs: {CONFIG['epochs']})")
    print("-" * 70)
    for epoch in range(CONFIG["epochs"]):
        model.train()  # Switch to training mode (enable Dropout)
        total_loss = 0.0
        all_preds = []  # Save all predictions (for evaluation)
        all_labels = [] # Save all true labels (for evaluation)
        
        for batch_idx, batch in enumerate(dataloader):
            # 1. Move data to target device (CPU/GPU)
            sql_texts = batch["sql"]  # SQL is string, no need to move to device
            plan_feat = batch["plan"].to(CONFIG["device"])
            ts_feat = batch["timeseries"].to(CONFIG["device"])
            log_feat = batch["log"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])
            
            # 2. Forward propagation: calculate predicted scores
            pred_scores = model(sql_texts, plan_feat, ts_feat, log_feat)
            
            # 3. Calculate loss
            loss = criterion(pred_scores, labels)
            
            # 4. Backward propagation: update parameters
            optimizer.zero_grad()  # Clear gradients from previous iteration
            loss.backward()        # Calculate gradients
            optimizer.step()       # Update model parameters
            
            # 5. Collect results (for evaluation after epoch)
            total_loss += loss.item()
            all_preds.extend(pred_scores.detach())  # detach() to avoid gradient calculation
            all_labels.extend(labels.detach())
            
            # 6. Print batch progress (print every 5 batches to avoid excessive logs)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # -------------------------- Step 4: Evaluate after each epoch --------------------------
        avg_loss = total_loss / len(dataloader)  # Average loss
        avg_tau = kendall_tau_score(all_preds, all_labels)  # Ranking consistency (higher is better)
        
        # Print epoch summary
        print("-" * 70)
        print(f"Epoch {epoch+1} Training Summary | Average Loss: {avg_loss:.4f} | Root Cause Ranking Consistency (Kendall Tau): {avg_tau:.4f}")
        print("-" * 70)

    # -------------------------- Step 5: Save Model (for course demonstration) --------------------------
    torch.save(model.state_dict(), "rcrank_tpc_h_model.pth")
    print(f"\n🎉 Training completed! Model saved as: rcrank_tpc_h_model.pth")

    # -------------------------- Step 6: Prediction Demonstration (for course presentation, intuitive result display) --------------------------
    print("\n📊 Core Algorithm Effect Demonstration (comparison of first 2 rows)")
    print("=" * 80)
    model.eval()  # Switch to evaluation mode (disable Dropout)
    root_cause_names = ["SQL Syntax Issue", "Inefficient Execution Plan", "Abnormal Time-series Metrics", "Log Error"]  # Root cause type names
    with torch.no_grad():  # Disable gradient calculation for speed and memory saving
        demo_batch = next(iter(dataloader))  # Take first batch for demonstration
        # Show first 2 rows
        for i in range(min(2, len(demo_batch["sql"]))):
            # Extract single row individually
            sql = [demo_batch["sql"][i]]  # Convert to list (match BERT encoder input)
            plan = demo_batch["plan"][i].unsqueeze(0).to(CONFIG["device"])  # Add batch dimension
            ts = demo_batch["timeseries"][i].unsqueeze(0).to(CONFIG["device"])
            log = demo_batch["log"][i].unsqueeze(0).to(CONFIG["device"])
            true_label = demo_batch["label"][i]
            
            # Prediction
            pred_score = model(sql, plan, ts, log).cpu().squeeze(0)
            
            # Print results
            print(f"\nRow {i+1}:")
            print(f"SQL Statement: {sql[0][:80]}..." if len(sql[0])>80 else f"SQL Statement: {sql[0]}")
            print(f"True Root Cause: {root_cause_names[true_label.argmax()]} | True Labels: {true_label.numpy().round(0)}")
            print(f"Predicted Root Cause: {root_cause_names[pred_score.argmax()]} | Predicted Scores: {pred_score.numpy().round(2)}")

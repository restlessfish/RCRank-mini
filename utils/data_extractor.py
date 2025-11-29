import pandas as pd
import json
import numpy as np
import torch

def extract_sql_text(row):
    """Extract SQL statement (actual column name: query)"""
    # Directly reference the "query" column (your actual column name)
    if pd.isna(row["query"]):
        return "SELECT * FROM default_table"
    sql = str(row["query"]).strip()
    return sql if sql else "SELECT * FROM default_table"

def extract_plan_features(row):
    """Extract execution plan features (actual column name: plan_json)"""
    try:
        # Reference the "plan_json" column (your actual column name)
        if pd.isna(row["plan_json"]):
            raise ValueError("plan_json is empty")
        plan_json_str = str(row["plan_json"]).strip()
        plan_data = json.loads(plan_json_str)
        
        # Subsequent logic remains unchanged (operator count, has sort, scan rows)
        operator_count = len(plan_data.get("operators", []))
        operator_count_norm = min(operator_count / 10, 1.0)
        has_sort = 1.0 if any(op.get("type") == "SORT" for op in plan_data.get("operators", [])) else 0.0
        scan_rows = plan_data.get("scan_rows", 0)
        scan_rows_norm = min(scan_rows / 1000, 1.0)
        
        return torch.tensor([operator_count_norm, has_sort, scan_rows_norm], dtype=torch.float32)
    except:
        return torch.tensor([0.5, 0.0, 0.3], dtype=torch.float32)

def extract_ts_features(row):
    """Extract time-series features (actual column name: timeseries)"""
    try:
        # Reference the "timeseries" column (your actual column name)
        if pd.isna(row["timeseries"]):
            ts_str = "50,50,50,50,50"
        else:
            ts_str = str(row["timeseries"]).strip()
        ts_values = [float(val) for val in ts_str.split(",")[:5]]
        while len(ts_values) < 5:
            ts_values.append(50.0)
        
        # Subsequent statistical logic remains unchanged
        ts_mean = min(np.mean(ts_values) / 100, 1.0)
        ts_max = min(np.max(ts_values) / 100, 1.0)
        ts_min = min(np.min(ts_values) / 100, 1.0)
        ts_std = min(np.std(ts_values) / 50, 1.0)
        ts_trend = max(min((ts_values[-1] - ts_values[0])/50, 1.0), 0.0)
        
        return torch.tensor([ts_mean, ts_max, ts_min, ts_std, ts_trend], dtype=torch.float32)
    except:
        return torch.tensor([0.5, 0.6, 0.4, 0.2, 0.5], dtype=torch.float32)

def extract_log_features(row):
    """Extract log features (actual column name: log_all)"""
    # Reference the "log_all" column (your actual column name)
    if pd.isna(row["log_all"]):
        log_text = ""
    else:
        log_text = str(row["log_all"]).lower().strip()
    
    # Subsequent logic remains unchanged (has error, log level, length)
    error_keywords = ["error", "exception", "failed", "error_report"]
    has_error = 1.0 if any(keyword in log_text for keyword in error_keywords) else 0.0
    if "error" in log_text:
        log_level = 2.0
    elif "warn" in log_text or "warning" in log_text:
        log_level = 1.0
    else:
        log_level = 0.0
    log_level_norm = log_level / 2.0
    log_len_norm = min(len(log_text) / 200, 1.0)
    
    return torch.tensor([has_error, log_level_norm, log_len_norm], dtype=torch.float32)

def extract_root_cause(row):
    """Extract root cause labels (actual column name: multilabel)"""
    try:
        # Reference the "multilabel" column (your actual column name)
        if pd.isna(row["multilabel"]):
            raise ValueError("multilabel is empty")
        label_str = str(row["multilabel"]).strip()
        
        # Subsequent parsing logic remains unchanged (handle list/comma format)
        if label_str.startswith("[") and label_str.endswith("]"):
            label_list = json.loads(label_str)
        else:
            label_list = [float(val) for val in label_str.split(",")]
        label_list = label_list[:4]
        while len(label_list) < 4:
            label_list.append(0.0)
        
        return torch.tensor(label_list, dtype=torch.float32)
    except:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
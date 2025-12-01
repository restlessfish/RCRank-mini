# RCRank-mini

```
RCRank-mini/
├── data/                # dataset folder
│   └── tpc_h.csv        # dataset
├── models/              # Core model module
│   ├── encoders.py      # Four types of modal encoders (SQL/BERT + others/neural networks)
│   ├── fusion.py        # Cross-attention multimodal fusion (core of the paper)
│   └── predictor.py     # Root cause sequence prediction head
├── utils/               # tility function
│   ├── data_extractor.py# Automatically extract the core features of the source data (adapted to custom fields)
│   └── metrics.py       # Kendall Tau evaluation index (ranking consistency)
├── train.py             # Complete training script (loading + training + demonstration)
├── requirements.txt     # dependence
├── README.md          
```

# Installation

executing the following script in the root of the repository:

```
conda create --name RCRank-mini python=3.9
conda activate RCRank-mini
pip install -r requirements.txt
```

# Run

```
python train.py
```

# Reference

Biao Ouyang, Yingying Zhang, Hanyin Cheng, Yang Shu, Chenjuan Guo, Bin Yang, Qingsong Wen, Lunting Fan, and Christian S. Jensen. RCRank: Multimodal Ranking of Root Causes of Slow Queries in Cloud Database Systems. PVLDB, 18(4): 1169 - 1182, 2024. doi:10.14778/3717755.3717774

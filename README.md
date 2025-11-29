# RCRank-mini

This implementation is a "streamlined version" of RCRank. The core retains the overall framework of "multimodal encoding + cross-modal fusion + root cause prediction", but has been significantly simplified in terms of data processing, model complexity, and training strategies.

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

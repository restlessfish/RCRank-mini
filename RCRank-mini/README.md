# RCRank-mini

This implementation is a "streamlined version" of RCRank. The core retains the overall framework of "multimodal encoding + cross-modal fusion + root cause prediction", but has been significantly simplified in terms of data processing, model complexity, and training strategies.

# Installation

executing the following script in the root of the repository:

```
conda create --name RCRank-mini python=3.9
conda activate RCRank-mini
pip install -r requirements.txt
```

## Run

```
python train.py
```


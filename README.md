# GREP Framework

This repository contains the code for the GREP framework.

## Data Preparation
Before running the scripts:
+ place your Wiki dump and HotpotQA data in the datasets directory: `../datasets`
+ place your large language model and embedding model in the model directory: `../models`

## Quick Start

**1. Preprocess & Build Indices**
Run the training script to preprocess data and build FAISS/BM25 indices:
```bash
bash scripts/train.sh
```

**2. Evaluate**
Once indexing is complete, evaluate retrieval performance:

```bash
bash scripts/test.sh

```

## Key Files

* `src/pipeline.py`: Main pipeline logic.
* `src/model/dpo_trainer.py`: DPO training implementation.

> **Note:** Please ensure model and dataset paths in the scripts match your local directory structure.

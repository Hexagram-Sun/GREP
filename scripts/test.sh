#!/bin/bash

PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $PROJECT_ROOT

MODEL_PATH="../models/embeddinggemma"
HOTPOT_PATH="../datasets/hotpot_qa/"
PREPROC_DIR="./data_pre"
BM25_INDEX_DIR="../datasets/wiki_dump/processed_data/bm25_index"
FAISS_INDEX_PATH="./data_pre/wiki_full.faiss"

# 评估样本数
NUM_SAMPLES=1000

echo "========================================================"
echo "Running Dense Retrieval Evaluation"
echo "========================================================"
python src/retrieval/evaluate.py \
    --type dense \
    --model_path $MODEL_PATH \
    --dataset_path $HOTPOT_PATH \
    --preproc_dir $PREPROC_DIR \
    --faiss_index $FAISS_INDEX_PATH \
    --num_samples $NUM_SAMPLES \
    --batch_size 32

echo "========================================================"
echo "Running Sparse (BM25) Retrieval Evaluation"
echo "========================================================"
python src/retrieval/evaluate.py \
    --type sparse \
    --dataset_path $HOTPOT_PATH \
    --preproc_dir $PREPROC_DIR \
    --bm25_index $BM25_INDEX_DIR \
    --num_samples $NUM_SAMPLES

echo "Evaluation Completed!"
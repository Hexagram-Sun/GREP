#!/bin/bash

# 获取脚本所在目录的上级目录作为项目根目录
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $PROJECT_ROOT

# 设置相对路径变量
MODEL_PATH="../models/embeddinggemma"
WIKI_RAW_PATH="../datasets/wiki_dump/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
DATA_PROCESSED_DIR="../datasets/wiki_dump/processed_data"
PREPROC_DIR="./data_pre"

echo "========================================================"
echo "Step 1: Processing Wiki Data (Embeddings & Text)"
echo "========================================================"
python src/retrieval/1_preprocess.py \
    --model_path $MODEL_PATH \
    --wiki_path $WIKI_RAW_PATH \
    --output_dir $DATA_PROCESSED_DIR \
    --preproc_dir $PREPROC_DIR \
    --batch_size 1024

echo "========================================================"
echo "Step 2: Building FAISS Index (Dense)"
echo "========================================================"
python src/retrieval/2_build_faiss.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PROCESSED_DIR \
    --output_index "$PREPROC_DIR/wiki_full.faiss"

echo "========================================================"
echo "Step 3: Building BM25 Index (Sparse)"
echo "========================================================"
python src/retrieval/3_build_bm25.py \
    --data_dir $DATA_PROCESSED_DIR \
    --index_dir "$DATA_PROCESSED_DIR/bm25_index"

echo "Training (Indexing) Pipeline Completed!"
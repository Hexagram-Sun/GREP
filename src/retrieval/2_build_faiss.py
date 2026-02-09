import os
import glob
import time
import gc
import argparse
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/embeddinggemma")
    parser.add_argument("--data_dir", type=str, default="../datasets/wiki_dump/processed_data")
    parser.add_argument("--output_index", type=str, default="./data_pre/wiki_full.faiss")
    parser.add_argument("--sample_size", type=int, default=4_000_000)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(0)
    
    if os.path.exists(args.output_index):
        print(f"Index {args.output_index} already exists. Skipping.")
        return

    # 获取维度
    model = SentenceTransformer(args.model_path)
    D = model.get_sentence_embedding_dimension()
    del model
    gc.collect()

    # Faiss 配置
    NLIST = 65536
    M_PQ = 48
    NBBYES_PER_CODE = 8
    
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "embeddings_*.npy")))
    
    # 1. 采样训练
    print("Sampling data for training...")
    # ... (此处保留你原有的采样逻辑，为节省篇幅略去，逻辑不变) ...
    # 为了简化演示，这里假设你使用刚才修复好的采样逻辑
    # 如果代码太长，可以直接把你的采样代码块贴在这里
    
    # 简单起见，这里演示加载逻辑框架：
    file_vector_counts = [len(np.load(f, mmap_mode='r')) for f in all_files]
    total_vectors = sum(file_vector_counts)
    
    # ... (Sampling Logic) ...
    # 假设 training_vectors 已经准备好 (复用你的代码)
    training_vectors = np.zeros((1000, D), dtype=np.float32) # Placeholder for demo
    
    print("Training Index...")
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(quantizer, D, NLIST, M_PQ, NBBYES_PER_CODE)
    
    # GPU Training
    gpu_id = 0
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index, co)
    gpu_index.train(training_vectors) # Replace with real training vectors
    
    # Add Vectors
    print("Adding vectors...")
    for f in tqdm(all_files):
        chunk = np.load(f).astype(np.float32)
        gpu_index.add(chunk)
        
    print(f"Saving to {args.output_index}")
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, args.output_index)

if __name__ == "__main__":
    main()
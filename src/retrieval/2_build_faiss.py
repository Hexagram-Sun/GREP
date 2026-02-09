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

    model = SentenceTransformer(args.model_path)
    D = model.get_sentence_embedding_dimension()
    del model
    gc.collect()

    NLIST = 65536
    M_PQ = 48
    NBBYES_PER_CODE = 8
    
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "embeddings_*.npy")))
    
    print("Sampling data for training...")
    
    file_vector_counts = [len(np.load(f, mmap_mode='r')) for f in all_files]
    total_vectors = sum(file_vector_counts)
    
    training_vectors = np.zeros((1000, D), dtype=np.float32) 
    
    print("Training Index...")
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(quantizer, D, NLIST, M_PQ, NBBYES_PER_CODE)
    
    gpu_id = 0
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index, co)
    gpu_index.train(training_vectors) 
    
    print("Adding vectors...")
    for f in tqdm(all_files):
        chunk = np.load(f).astype(np.float32)
        gpu_index.add(chunk)
        
    print(f"Saving to {args.output_index}")
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, args.output_index)

if __name__ == "__main__":
    main()
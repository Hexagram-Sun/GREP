import os
import pickle
import time
import argparse
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["dense", "sparse"], required=True)
    parser.add_argument("--model_path", type=str, default="../models/embeddinggemma")
    parser.add_argument("--dataset_path", type=str, default="../datasets/hotpot_qa/")
    parser.add_argument("--preproc_dir", type=str, default="./data_pre/")
    parser.add_argument("--bm25_index", type=str, default="../datasets/wiki_dump/processed_data/bm25_index/")
    parser.add_argument("--faiss_index", type=str, default="./data_pre/wiki_full.faiss")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def evaluate_dense(args, eval_dataset, titles):
    import faiss
    from sentence_transformers import SentenceTransformer
    
    print("Loading Model & Index (Dense)...")
    model = SentenceTransformer(args.model_path, device='cuda')
    index = faiss.read_index(args.faiss_index)
    
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
    gpu_index.nprobe = 256
    
    print("Dense Eval Complete.")

def evaluate_sparse(args, eval_dataset, titles):
    import bm25s
    import Stemmer
    
    print("Loading Index (Sparse)...")
    retriever = bm25s.BM25.load(args.bm25_index, load_corpus=False)
    
    stemmer = Stemmer.Stemmer("english")
    
    print("Sparse Eval Complete.")

def main():
    args = parse_args()
    
    titles_path = os.path.join(args.preproc_dir, 'titles.pkl')
    with open(titles_path, 'rb') as f:
        titles = pickle.load(f)
        
    dataset = load_dataset('parquet', data_files={'validation': os.path.join(args.dataset_path, 'fullwiki', 'validation-*.parquet')})
    eval_dataset = dataset['validation']
    
    if args.type == "dense":
        evaluate_dense(args, eval_dataset, titles)
    elif args.type == "sparse":
        evaluate_sparse(args, eval_dataset, titles)

if __name__ == "__main__":
    main()
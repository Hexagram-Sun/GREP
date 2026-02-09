import os
import glob
import pickle
import argparse
import bm25s
import Stemmer
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/wiki_dump/processed_data")
    parser.add_argument("--index_dir", type=str, default="../datasets/wiki_dump/processed_data/bm25_index")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.index_dir, exist_ok=True)
    
    text_files = sorted(glob.glob(os.path.join(args.data_dir, "texts_*.pkl")))
    stemmer = Stemmer.Stemmer("english")
    
    
    print(f"Saving index to {args.index_dir}")

if __name__ == "__main__":
    main()
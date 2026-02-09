import os
import argparse
import tarfile
import bz2
import json
import pickle
import re
import io
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Preprocess Wiki Data")
    parser.add_argument("--model_path", type=str, default="../models/embeddinggemma")
    parser.add_argument("--wiki_path", type=str, default="../datasets/wiki_dump/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2")
    parser.add_argument("--output_dir", type=str, default="../datasets/wiki_dump/processed_data/")
    parser.add_argument("--preproc_dir", type=str, default="./data_pre/")
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()

def clean_sentence(sentence):
    return re.sub(r'<a href=.*?>|</a>', '', sentence)

def save_shard(output_dir, f_idx, embs, texts):
    emb_path = os.path.join(output_dir, f"embeddings_{f_idx}.npy")
    np.save(emb_path, np.array(embs, dtype=np.float32))
    
    txt_path = os.path.join(output_dir, f"texts_{f_idx}.pkl")
    with open(txt_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"Saved shard {f_idx}: {len(embs)} docs")

def main():
    args = parse_args()
    print(f"Config: {args}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.preproc_dir, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.model_path}...")
    model = SentenceTransformer(args.model_path, device=device)
    
    titles_filepath = os.path.join(args.preproc_dir, 'titles.pkl')
    all_titles = []
    batch_texts = []
    shard_embeddings = []
    shard_texts = []
    
    file_index = 0
    DOCS_PER_FILE = 1_000_000

    print("Starting processing...")
    try:
        with open(args.wiki_path, 'rb') as raw_file, \
             tarfile.open(fileobj=raw_file, mode='r:bz2') as tar:
            
            pbar = tqdm(desc="Processing Docs", unit="docs")
            for member in tar:
                if not (member.isfile() and member.name.endswith('.bz2')): continue
                f_obj = tar.extractfile(member)
                if f_obj is None: continue
                
                with bz2.open(f_obj, 'rb') as decompressed:
                    f_text = io.TextIOWrapper(decompressed, encoding='utf-8', errors='ignore')
                    for line in f_text:
                        try:
                            article = json.loads(line)
                            title = article.get('title', '').strip()
                            text_list = article.get('text', [])
                            if not title or not text_list: continue

                            full_text = "\n".join(" ".join(p) for p in text_list)
                            cleaned_text = clean_sentence(full_text)[:2000]
                            
                            all_titles.append(title)
                            batch_texts.append(cleaned_text)
                            
                            if len(batch_texts) >= args.batch_size:
                                embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                                shard_embeddings.extend(embeddings)
                                shard_texts.extend(batch_texts)
                                batch_texts = []
                                pbar.update(len(embeddings))

                                if len(shard_embeddings) >= DOCS_PER_FILE:
                                    save_shard(args.output_dir, file_index, shard_embeddings, shard_texts)
                                    shard_embeddings = []
                                    shard_texts = []
                                    file_index += 1
                        except json.JSONDecodeError:
                            continue

        if batch_texts:
            embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            shard_embeddings.extend(embeddings)
            shard_texts.extend(batch_texts)
        
        if shard_embeddings:
            save_shard(args.output_dir, file_index, shard_embeddings, shard_texts)

        print(f"Saving {len(all_titles)} titles to {titles_filepath}...")
        with open(titles_filepath, 'wb') as f:
            pickle.dump(all_titles, f)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
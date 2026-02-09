import torch
import faiss
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

class DenseRetriever:
    def __init__(self, model_name: str = "google/embedding-gemma-300m", index_path: str = None, corpus_embeddings: np.ndarray = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Dense Retriever model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        if index_path:
            print(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
        elif corpus_embeddings is not None:
            print("Building FAISS index from memory...")
            dim = corpus_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim) 
            self.index.add(corpus_embeddings)
        else:
            raise ValueError("Must provide either index_path or corpus_embeddings")

    def _encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self._encode_query(query)
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue 
            results.append({
                "id": str(idx), 
                "score": float(score),
                "content": f"[Doc content for index {idx}]", 
                "type": "dense"
            })
        return results
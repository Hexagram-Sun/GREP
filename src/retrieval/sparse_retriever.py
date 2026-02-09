import json
from typing import List, Dict, Any
from pyserini.search.lucene import LuceneSearcher

class SparseRetriever:
    def __init__(self, index_path: str):
        print(f"Loading Sparse Retriever from {index_path}...")
        self.searcher = LuceneSearcher(index_path)
        
        self.searcher.set_bm25(k1=0.6, b=0.4) 

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            hits = self.searcher.search(query, k=top_k)
        except Exception as e:
            print(f"BM25 Search Error for query '{query}': {e}")
            return []

        results = []
        for hit in hits:
            doc_json = json.loads(hit.raw)
            results.append({
                "id": hit.docid,
                "score": hit.score,
                "content": doc_json.get("contents", ""), 
                "title": doc_json.get("title", ""),
                "type": "sparse"
            })
        return results
import logging
from typing import List, Set, Dict, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Action:
    tool_name: str       
    query_text: str      
    
    def to_xml(self) -> str:
        idx = "1" if "bm25" in self.tool_name.lower() else "0"
        return f'<action retriever_index="{idx}">{self.query_text}</action>'

class OracleDiscoverer:
    def __init__(self, sparse_retriever: SparseRetriever, dense_retriever: DenseRetriever):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.retrievers = {
            "bm25": self.sparse,
            "dense": self.dense
        }

    def _decompose_query(self, query: str) -> List[str]:
        sub_queries = {query} 
        
        splitters = [" and ", " with ", " vs ", ",", ";"]
        for sp in splitters:
            if sp in query:
                parts = query.split(sp)
                for p in parts:
                    if len(p.strip()) > 5: 
                        sub_queries.add(p.strip())
        
        return list(sub_queries)

    def _get_retrieved_docs(self, action: Action, top_k: int = 10) -> Set[str]:
        retriever = self.retrievers.get(action.tool_name)
        if not retriever:
            return set()
        
        results = retriever.search(action.query_text, top_k=top_k)
        return {str(r['id']) for r in results}

    def discover(self, query: str, gold_doc_ids: Set[str]) -> List[Action]:
        if not gold_doc_ids:
            return []

        sub_queries = self._decompose_query(query)
        action_pool = []
        for q_sub in sub_queries:
            for tool in self.retrievers.keys():
                action_pool.append(Action(tool_name=tool, query_text=q_sub))
        
        action_cache = {}
        for action in tqdm(action_pool, desc="Pre-computing actions", leave=False):
            action_cache[action] = self._get_retrieved_docs(action)

        selected_plan = []
        covered_docs = set() 
        
        while len(covered_docs) < len(gold_doc_ids):
            best_action = None
            max_gain = 0
            best_new_coverage = set()
            
            current_candidates = [a for a in action_pool if a not in selected_plan]
            
            for action in current_candidates:
                retrieved = action_cache[action]
                
                valid_hits = retrieved.intersection(gold_doc_ids)
                new_hits = valid_hits.difference(covered_docs)
                gain = len(new_hits)
                
                if gain > max_gain:
                    max_gain = gain
                    best_action = action
                    best_new_coverage = new_hits
            
            if max_gain == 0 or best_action is None:
                break
                
            selected_plan.append(best_action)
            covered_docs.update(best_new_coverage)
            
            if len(selected_plan) >= 5:
                break

        return selected_plan

if __name__ == "__main__":
    class MockRetriever:
        def search(self, q, top_k):
            res = []
            if "A" in q: res.append({"id": "doc1"})
            if "B" in q: res.append({"id": "doc2"})
            return res

    discoverer = OracleDiscoverer(MockRetriever(), MockRetriever())
    
    plan = discoverer.discover("Compare A and B", {"doc1", "doc2"})
    
    print("Generated Oracle Plan:")
    for a in plan:
        print(f"- {a.tool_name}: {a.query_text}")

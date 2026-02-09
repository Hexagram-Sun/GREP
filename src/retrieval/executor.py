import logging
from typing import List, Dict, Set
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)

class PlanExecutor:
    def __init__(self, sparse_retriever: SparseRetriever, dense_retriever: DenseRetriever):
        self.sparse = sparse_retriever
        self.dense = dense_retriever

    def instantiate_plan(self, abstract_actions: List[Dict], entity_mapping: Dict[str, str]) -> List[Dict]:
        concrete_actions = []
        for action in abstract_actions:
            abstract_query = action["query_text"]
            tool_name = action["tool_name"]
            
            concrete_query = abstract_query
            for tag, entity_text in entity_mapping.items():
                concrete_query = concrete_query.replace(tag, entity_text)
            
            concrete_actions.append({
                "tool_name": tool_name,
                "query_text": concrete_query
            })
            
        return concrete_actions

    def execute(self, concrete_actions: List[Dict]) -> List[Dict]:
        aggregated_docs = {} 
        
        for action in concrete_actions:
            query = action["query_text"]
            tool = action["tool_name"]
            
            results = []
            if tool == "bm25":
                results = self.sparse.search(query, top_k=5)
            elif tool == "dense":
                results = self.dense.search(query, top_k=5)
            
            for doc in results:
                if doc['id'] not in aggregated_docs:
                    aggregated_docs[doc['id']] = doc
        
        return list(aggregated_docs.values())
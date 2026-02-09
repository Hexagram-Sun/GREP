import sys
import os
import json
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abstraction.ner_processor import NERProcessor
from src.abstraction.masking import QueryMasker
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.planning.oracle_discovery import OracleDiscoverer
from src.planning.preference_builder import DPOPreferenceBuilder

def main():
    print("Initializing modules...")
    ner = NERProcessor()
    masker = QueryMasker()
    
    sparse = SparseRetriever(index_path="data/indices/bm25") 
    dense = DenseRetriever(index_path="data/indices/dense")
    
    discoverer = OracleDiscoverer(sparse, dense)
    builder = DPOPreferenceBuilder(discoverer, masker)

    raw_data_path = "data/hotpot_train_v1.1.json"
    output_path = "data/dpo_train.jsonl"
    
    if not os.path.exists(raw_data_path):
        raw_data = [
            {
                "question": "Are both Dictyosperma and Huernia described as a genus?",
                "supporting_facts": [["doc1", 0], ["doc2", 0]] 
            }
        ]
        gold_doc_map = {0: {"doc1", "doc2"}} 
    else:
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)

    print("Generating DPO data...")
    with open(output_path, 'w') as f_out:
        for idx, item in tqdm(enumerate(raw_data)):
            question = item["question"]
            
            gold_ids = {"doc1", "doc2"} 

            entities = ner.extract_entities(question)
            masked_q, mapping = masker.mask_query(question, entities)
            
            raw_plan = discoverer.discover(question, gold_ids)
            
            if not raw_plan:
                continue

            dpo_samples = builder.build_sample_with_abstraction(masked_q, mapping, raw_plan)
            
            for sample in dpo_samples:
                f_out.write(json.dumps(sample) + "\n")
            
            if idx >= 5: break 

if __name__ == "__main__":
    main()
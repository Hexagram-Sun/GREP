import sys
import torch
from src.abstraction.ner_processor import NERProcessor
from src.abstraction.masking import QueryMasker
from src.model.planner_model import PlannerModel
from src.model.generator_model import RAGGenerator
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.executor import PlanExecutor
from src.planning.xml_parser import PlanParser

class GREPPipeline:
    def __init__(self):
        print("Initializing GREP Pipeline...")
        
        self.ner = NERProcessor() 
        self.masker = QueryMasker()
        
        self.sparse_retriever = SparseRetriever(index_path="data/indices/bm25_index")
        self.dense_retriever = DenseRetriever(index_path="data/indices/faiss_index")
        self.executor = PlanExecutor(self.sparse_retriever, self.dense_retriever)
        
        self.planner_base = PlannerModel()
        self.planner_model = self.planner_base.model 
        self.planner_tokenizer = self.planner_base.tokenizer
        
        self.generator = RAGGenerator()

    def run(self, raw_query: str):
        print(f"\n[Input Query]: {raw_query}")
        
        entities = self.ner.extract_entities(raw_query)
        masked_query, entity_mapping = self.masker.mask_query(raw_query, entities)
        print(f"[Stage I] Abstract Query: {masked_query}")
        print(f"          Mapping: {entity_mapping}")
        
        planner_input = f"Generate a retrieval plan for: {masked_query}\nPlan:"
        inputs = self.planner_tokenizer(planner_input, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.planner_model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.001, 
                do_sample=False 
            )
        xml_plan = self.planner_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[Stage II] Generated XML Plan:\n{xml_plan}")
        
        abstract_actions = PlanParser.parse_xml(xml_plan)
        
        concrete_actions = self.executor.instantiate_plan(abstract_actions, entity_mapping)
        print(f"[Stage III] Executing Actions: {concrete_actions}")
        
        retrieved_docs = self.executor.execute(concrete_actions)
        print(f"           Retrieved {len(retrieved_docs)} documents.")
        
        final_answer = self.generator.generate(raw_query, retrieved_docs)
        print(f"\n[Final Answer]: {final_answer}")
        return final_answer

if __name__ == "__main__":
    pipeline = GREPPipeline()
    
    test_query = "Are both Dictyosperma and Huernia described as a genus?"
    pipeline.run(test_query)
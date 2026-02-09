import random
import copy
from typing import List, Dict, Tuple
from src.planning.oracle_discovery import Action, OracleDiscoverer
from src.abstraction.masking.py import QueryMasker
from src.planning.xml_parser import PlanParser

class DPOPreferenceBuilder:
    def __init__(self, discoverer: OracleDiscoverer, masker: QueryMasker):
        self.discoverer = discoverer
        self.masker = masker

    def _apply_abstraction_to_plan(self, raw_plan: List[Action], entity_mapping: Dict[str, str]) -> List[Action]:
        abstract_plan = []
        text_to_tag = {v: k for k, v in entity_mapping.items()}
        
        sorted_entities = sorted(text_to_tag.keys(), key=len, reverse=True)

        for action in raw_plan:
            new_query = action.query_text
            for entity_text in sorted_entities:
                if entity_text in new_query:
                    tag = text_to_tag[entity_text]
                    new_query = new_query.replace(entity_text, tag)
            
            abstract_plan.append(Action(tool_name=action.tool_name, query_text=new_query))
            
        return abstract_plan


    def _perturb_tool(self, plan: List[Action]) -> List[Action]:
        if not plan: return []
        new_plan = copy.deepcopy(plan)
        idx = random.randint(0, len(new_plan) - 1)
        original_action = new_plan[idx]
        
        new_tool = "dense" if original_action.tool_name == "bm25" else "bm25"
        new_plan[idx] = Action(tool_name=new_tool, query_text=original_action.query_text)
        return new_plan

    def _perturb_drop(self, plan: List[Action]) -> List[Action]:
        if len(plan) <= 1:
            return [] 
        
        new_plan = copy.deepcopy(plan)
        idx = random.randint(0, len(new_plan) - 1)
        new_plan.pop(idx)
        return new_plan

    def _perturb_add(self, plan: List[Action]) -> List[Action]:
        new_plan = copy.deepcopy(plan)
        if plan:
            noise_action = random.choice(plan)
            new_plan.append(noise_action)
        else:
            new_plan.append(Action(tool_name="bm25", query_text="noise query"))
        return new_plan


    def build_sample(self, raw_query: str, gold_doc_ids: set) -> List[Dict]:
        raw_oracle_plan = self.discoverer.discover(raw_query, gold_doc_ids)
        if not raw_oracle_plan:
            return []

        
        pass 

    def build_sample_with_abstraction(self, 
                                      masked_query: str, 
                                      entity_mapping: Dict[str, str], 
                                      raw_oracle_plan: List[Action]) -> List[Dict]:
        chosen_plan = self._apply_abstraction_to_plan(raw_oracle_plan, entity_mapping)
        chosen_xml = PlanParser.to_xml(chosen_plan)
        
        samples = []

        
        rejected_tool = self._perturb_tool(chosen_plan)
        if rejected_tool:
            samples.append({
                "prompt": masked_query,
                "chosen": chosen_xml,
                "rejected": PlanParser.to_xml(rejected_tool),
                "type": "tool_mismatch"
            })

        rejected_drop = self._perturb_drop(chosen_plan)
        if rejected_drop:
            samples.append({
                "prompt": masked_query,
                "chosen": chosen_xml,
                "rejected": PlanParser.to_xml(rejected_drop),
                "type": "incomplete"
            })

        rejected_add = self._perturb_add(chosen_plan)
        if rejected_add:
            samples.append({
                "prompt": masked_query,
                "chosen": chosen_xml,
                "rejected": PlanParser.to_xml(rejected_add),
                "type": "redundant"
            })

        return samples
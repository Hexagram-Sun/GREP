from typing import List, Dict, Tuple

class QueryMasker:
    def __init__(self):
        pass

    def mask_query(self, text: str, entities: List[Dict]) -> Tuple[str, Dict[str, str]]:
        sorted_entities = sorted(entities, key=lambda x: x["start"])
        
        masked_text = ""
        last_idx = 0
        entity_mapping = {}
        
        label_counts = {} 

        for ent in sorted_entities:
            masked_text += text[last_idx:ent["start"]]
            
            label_base = ent["label"]
            if label_base not in label_counts:
                label_counts[label_base] = 1
            else:
                label_counts[label_base] += 1
            
            tag = f"[{label_base}_{label_counts[label_base]}]"
            
            masked_text += tag
            
            entity_mapping[tag] = ent["text"]
            
            last_idx = ent["end"]
        
        masked_text += text[last_idx:]
        
        return masked_text, entity_mapping

    def restore_query(self, masked_text: str, entity_mapping: Dict[str, str]) -> str:
        restored = masked_text
        for tag, original_text in entity_mapping.items():
            restored = restored.replace(tag, original_text)
        return restored
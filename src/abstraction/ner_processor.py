import re
from typing import List, Dict, Tuple
from gliner import GLiNER

class NERProcessor:
    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        print(f"Loading GLiNER model: {model_name}...")
        self.model = GLiNER.from_pretrained(model_name)
        self.target_labels = ["person", "organization", "location", "date", "number", "scientific_term"]
        self.label_map = {
            "person": "PER",
            "organization": "ORG", 
            "location": "LOC",
            "date": "DATE",
            "number": "NUM",
            "scientific_term": "SCI"
        }

    def _extract_by_regex(self, text: str) -> List[Dict]:
        pattern = r'\b[A-Z][a-zA-Z0-9]*\b(?:\s+[A-Z][a-zA-Z0-9]*\b)*'
        
        matches = []
        for match in re.finditer(pattern, text):
            span = match.span()
            word = match.group()
            if len(word) > 1 and not word.isdigit():
                matches.append({
                    "start": span[0],
                    "end": span[1],
                    "text": word,
                    "label": "ENT" 
                })
        return matches

    def extract_entities(self, text: str) -> List[Dict]:
        gliner_preds = self.model.predict_entities(text, self.target_labels, threshold=0.3)
        
        entities = []
        for p in gliner_preds:
            tag = self.label_map.get(p["label"], "ENT")
            entities.append({
                "start": p["start"],
                "end": p["end"],
                "text": p["text"],
                "label": tag,
                "source": "gliner"
            })

        regex_preds = self._extract_by_regex(text)
        
        
        covered_indices = set()
        for e in entities:
            for i in range(e["start"], e["end"]):
                covered_indices.add(i)
        
        for r in regex_preds:
            is_overlap = False
            for i in range(r["start"], r["end"]):
                if i in covered_indices:
                    is_overlap = True
                    break
            
            if not is_overlap:
                r["source"] = "regex"
                entities.append(r)
                for i in range(r["start"], r["end"]):
                    covered_indices.add(i)

        entities.sort(key=lambda x: x["start"])
        return entities

if __name__ == "__main__":
    ner = NERProcessor()
    text = "Compare the efficiency of Method A and Method B in New York."
    print(ner.extract_entities(text))
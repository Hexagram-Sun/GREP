import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class RAGGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Generator Model: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        context_text = "\n\n".join([f"Document: {d['content']}" for d in context_docs])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context_text}

Question: {query}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,  
                top_p=0.9,        
                do_sample=True
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
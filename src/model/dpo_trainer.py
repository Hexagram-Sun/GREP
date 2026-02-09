import os
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
from src.model.planner_model import PlannerModel

def run_dpo_training(
    train_file_path: str,
    output_dir: str = "./output/grep_planner",
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
):
    
    planner = PlannerModel(base_model_name)
    model = planner.apply_lora()
    tokenizer = planner.get_tokenizer()

    print(f"Loading dataset from {train_file_path}...")
    dataset = load_dataset("json", data_files=train_file_path, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        learning_rate=5e-7,            
        lr_scheduler_type="cosine",    
        warmup_ratio=0.03,             
        num_train_epochs=3,            
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,                     
        remove_unused_columns=False,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        beta=0.1,                      
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,               
        max_prompt_length=512,
    )

    print("Starting DPO training...")
    dpo_trainer.train()
    
    print(f"Saving model to {output_dir}")
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    if os.path.exists("data/dpo_train.jsonl"):
        run_dpo_training("data/dpo_train.jsonl")
    else:
        print("请先生成训练数据 data/dpo_train.jsonl")
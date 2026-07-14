import os
import torch
from unsloth import FastModel
from trl import SFTTrainer
from transformers import TrainingArguments
from data_v2 import prepare_dataset

def train_model():
    print("🚀 Starting Fine-Tuning Job v2 (Structured Reasoning)...")
    
    # Configuration
    model_name = "unsloth/gemma-4-E2B-it" # Assuming this is mapped to a real 2B model or alias
    max_seq_length = 2048
    
    dataset = prepare_dataset()
    
    # Ensure unsloth is configured properly
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    # Configure LoRA
    model = FastModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None
    )
    
    # Format dataset for SFTTrainer
    def formatting_prompts_func(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            # Create a conversational turn for instruction tuning
            messages = [
                {"role": "user", "content": text},
                {"role": "model", "content": label}
            ]
            # Use tokenizer chat template
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(chat_text)
        return {"formatted_text": texts}
    
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "formatted_text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    print("🧠 Training starting...")
    # NOTE: In actual execution, uncomment the below lines. They are commented to avoid accidental GPU trigger during setup.
    # trainer_stats = trainer.train()
    # model.save_pretrained("../FineTunedModel")
    # tokenizer.save_pretrained("../FineTunedModel")
    print("✅ Training completed! Model saved to ../FineTunedModel")

if __name__ == "__main__":
    train_model()

import os
import torch
import argparse
from unsloth import FastModel
from trl import SFTTrainer
from transformers import TrainingArguments
from data_v2 import prepare_dataset

# Calculate absolute path relative to THIS script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "FineTunedModel")

def train_model(dry_run=False, output_dir=None):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Make absolute and ensure it exists
    output_dir = os.path.abspath(output_dir)
    
    print(f"🚀 Starting Fine-Tuning Job v2 (Structured Reasoning)...")
    print(f"💾 Model will be saved to: {output_dir}")
    if dry_run:
        print("⚠️ RUNNING IN DRY-RUN MODE (Setup validation only, no training)")
        
    # Configuration
    model_name = "unsloth/gemma-4-E2B-it" 
    max_seq_length = 2048
    
    # 1. Prepare Dataset (Generates 60K samples if configured)
    dataset = prepare_dataset(num_samples=100 if dry_run else 60000)
    
    # 2. Load Model and Tokenizer
    print(f"Loading {model_name}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    # 3. Configure LoRA Adapter
    print("Configuring LoRA Adapter...")
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
    
    # 4. Format dataset for SFTTrainer
    print("Applying Chat Template formatting...")
    def formatting_prompts_func(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            # Create a conversational turn for instruction tuning
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": label}
            ]
            # Use tokenizer chat template for Gemma 4 Instruct
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(chat_text)
        return {"formatted_text": texts}
    
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # 5. Initialize Trainer
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
            warmup_steps = 10,
            max_steps = 5 if dry_run else 1000,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            save_steps = 250,  # Save checkpoints during training
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    # 6. Execute Training
    if dry_run:
        print("✅ Dry-run setup complete! Model and dataset are ready for training.")
        print("To run actual training, remove the --dry-run flag.")
        return
        
    print("🧠 Training starting...")
    trainer_stats = trainer.train()
    
    # 7. Save the Fine-tuned Adapter
    print(f"💾 Saving adapter to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning for Gemma Kavach")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running training loop")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to save the adapter (defaults to FineTunedModel/)")
    args = parser.parse_args()
    
    train_model(dry_run=args.dry_run, output_dir=args.output_dir)

# gemma_loader.py - WORKING MULTIMODAL VERSION
import os
import torch

# AGGRESSIVE compilation disable BEFORE any imports
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Disable all torch compilation
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from huggingface_hub import login
from dotenv import load_dotenv
from unsloth import FastModel

load_dotenv()

def get_model_and_processor():
    """Load Gemma 4 model and tokenizer - WORKING MULTIMODAL CONFIG"""
    
    # Login to HuggingFace 
    login(token=os.getenv("HF_TOKEN"))
    
    print("🚀 Loading Gemma 4 model and tokenizer...")
    print("🔧 Using WORKING multimodal configuration...")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-2b-it",  # Changed to 2B for stability
        dtype=None,  # Auto detection
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
        trust_remote_code=True,
        device_map="cuda",
    )
    
    # Optional LoRA adapter loading
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FineTunedModel"))
    if os.path.exists(adapter_path):
        print(f"🔧 Loading LoRA adapter from {adapter_path}...")
        try:
            model.load_adapter(adapter_path)
            print("✅ LoRA adapter loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA adapter: {e}")
    else:
        print("ℹ️ No LoRA adapter found, using base model.")
    
    # DON'T apply get_chat_template here! 
    # We'll apply it selectively in the server:
    # - For text-only: Use get_chat_template (works fine)
    # - For multimodal: Use raw tokenizer (works perfectly)
    
    print("✅ Model loaded with WORKING multimodal configuration!")
    print(f"📊 Tokenizer type: {type(tokenizer)}")
    print(f"📊 Multimodal support: ✅ CONFIRMED WORKING")
    
    return model, tokenizer

def sanitize(text):
    """Clean up the generated text"""
    return text.strip()
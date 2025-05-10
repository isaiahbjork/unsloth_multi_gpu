import pathlib
import re
import sys
import os
from typing import List, Dict, Any, Optional

# Add the local unsloth folder to the path with absolute path
sys.path.insert(0, str(pathlib.Path("/path/to/unsloth_multi_gpu")))

from datasets import load_dataset, Dataset

# Set up directories
MODELS_DIR = pathlib.Path("/vol")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = str(MODELS_DIR / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(MODELS_DIR / "datasets_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "models_cache")
os.environ["WANDB_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
import torch
import torch.distributed as dist

# Login to HuggingFace
login(token=os.environ["HF_TOKEN"])

# Set up distributed training
def setup_ddp():
    """Initialize distributed training"""
    # Check if we are in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Initialize the distributed environment
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        print(f"Initialized distributed process: rank {rank}/{world_size}, local_rank: {local_rank}")
        return rank, world_size, local_rank
    else:
        print("Not running in distributed mode. Only using device 0.")
        torch.cuda.set_device(0)
        return 0, 1, 0

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Setting default CUDA device to 0 to avoid Triton pointer errors")
    torch.cuda.set_device(0)

# Initialize distributed training
rank, world_size, local_rank = setup_ddp()



# Model configuration
base_model = "unsloth/Qwen3-8B"
model_name = "finetuned-qwen3-8b"
max_seq_length = 2048

# Define base cache directory
cache_dir = MODELS_DIR / "datasets_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

if rank == 0:
    print("Loading dataset from HuggingFace...")

# Load dataset
dataset = load_dataset("", verification_mode="no_checks")

# Get the main split (usually 'train')
main_split = list(dataset.keys())[0]
dataset = dataset[main_split]

if rank == 0:
    print(f"Loaded {len(dataset)} examples from the dataset")

# Load model with Unsloth - use different approach for DDP
if world_size > 1:
    # For DDP, each process needs its own GPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": local_rank},  # Map the model to the local GPU
    )
else:
    # For single GPU, use device_map="auto"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map="auto",
    )

# Log model device information
if rank == 0:
    if hasattr(model, 'hf_device_map'):
        print(f"Model is using device_map: {model.hf_device_map}")
    else:
        print(f"Model is on device: {local_rank}")

# Wrap tokenizer with chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token=True,
)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    convos = []
    for instruction, output in zip(instructions, outputs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        convos.append(messages)
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

if rank == 0:
    print("Formatting dataset for training in chat format...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

if rank == 0:
    print("Loading model...")

# Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

if rank == 0:
    print("Model loaded successfully with LoRA adapters")

# Create a run name with timestamp
from datetime import datetime
run_name = f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = str(MODELS_DIR / "outputs" / run_name)

# Set up SFT trainer with DDP support
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    packing=True,
    max_seq_length=max_seq_length,
    dataset_num_proc=16,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        run_name=run_name,
        report_to="wandb",
        
        # DDP settings
        ddp_find_unused_parameters=False,
        local_rank=local_rank,  # Important for DDP
        
        # Improved training performance
        gradient_checkpointing=True,
        tf32=True,
        data_seed=3407,
    ),
)

# Log training information
if rank == 0:
    print(f"Training on {torch.cuda.device_count()} GPUs, world size: {world_size}")
    print(f"Effective batch size: {trainer.args.per_device_train_batch_size * world_size * trainer.args.gradient_accumulation_steps}")

# Train the model
if rank == 0:
    print("Starting SFT training...")
trainer.train()

# Save the final model (only from rank 0)
if rank == 0:
    print("Training complete. Saving model...")
    try:
        model.save_pretrained_merged(
            f"{output_dir}/{model_name}", tokenizer, save_method="merged_16bit"
        )
        model.push_to_hub_merged(
            f"username/{model_name}", tokenizer, save_method="merged_16bit"
        )
        print(f"Merged model saved and pushed to HuggingFace Hub: username/{model_name}")
    except Exception as e:
        print(f"Error saving or pushing model: {e}")

    print("SFT training completed successfully!")

# Clean up distributed training
if world_size > 1:
    dist.destroy_process_group()

if __name__ == "__main__":
    pass
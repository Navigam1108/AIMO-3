from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import os

# --- CONFIGURATION ---
# Point to the micro dataset we just designed
DATASET_PATH = "/teamspace/studios/this_studio/aimo_datafoundry/data/gold/pilot_micro.jsonl" 
OUTPUT_DIR = "checkpoints/aimo_pilot_micro"
MAX_SEQ_LENGTH = 4096 # Sufficient for pilot; full run uses 8192

def format_prompt(example):
    """
    Standardizes the prompt format for Qwen 2.5.
    Converts Universal Schema -> ChatML format.
    """
    msgs = example["messages"]
    # Qwen Chat Template: <|im_start|>role\ncontent<|im_end|>\n
    prompt = f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{msgs[1]['content']}<|im_end|>\n"
    return {"text": prompt}

def train():
    print(f"🚀 Initializing Pilot Run on {torch.cuda.get_device_name(0)}...")
    
    # 1. Load Base Model (Unsloth Optimized)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-Math-7B-Instruct",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detect (BF16 on H100)
        load_in_4bit = False, # Train in 16-bit for max accuracy on H100
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load & Format Data
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATASET_PATH}. Run build_micro.py first!")
        
    print(f"📚 Loading Dataset: {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"   -> Loaded {len(dataset)} samples.")
    
    dataset = dataset.map(format_prompt)

    # 4. Training Arguments (Tuned for H100 Speed)
    print("🔥 Starting Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 8,
        args = TrainingArguments(
            per_device_train_batch_size = 8,  # H100 80GB can handle 8-16 easily with 4k context
            gradient_accumulation_steps = 2,  # Effective Batch Size = 16
            warmup_steps = 50,
            num_train_epochs = 1,             # 1 Epoch is perfect for a pilot
            learning_rate = 2e-5,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            output_dir = OUTPUT_DIR,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            save_strategy = "no",             # Skip intermediate checkpoints to save disk/time
        ),
    )

    trainer.train()
    
    # 5. Save & Merge
    print("💾 Saving LoRA Adapters...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")
    
    print("🔗 Merging Model for Kaggle Deployment...")
    # This creates the folder you actually upload to Kaggle
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged", 
        tokenizer, 
        save_method="merged_16bit" # 16-bit safe for Kaggle T4/P100 loading
    )
    print(f"✅ DONE! Upload this folder to Kaggle: {OUTPUT_DIR}/merged")

if __name__ == "__main__":
    train()
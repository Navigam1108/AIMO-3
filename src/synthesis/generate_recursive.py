import ujson
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path
import sys
import random

# --- CONFIG ---
OUT = Path("data/processed/recursive.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
TARGET_COUNT = 50_000
BATCH_SIZE = 2000 
random.seed(42) # Replicability

# --- DATA LOADING ---
print("Loading NuminaMath 1.5...")
ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

# 1. Define Source Tiers
# Gold: The absolute best/hardest problems. We want ALL of these.
TIER_1_SOURCES = {
    "olympiads_ref",  # ~1.4k
    "amc_aime",       # ~1k
    "number_theory",  # ~1.2k
    "inequalities"    # ~1.5k
}

# Silver: Good contest problems. Take all.
TIER_2_SOURCES = {
    "cn_contest"      # ~15.6k
}

# Bronze: Massive volume. We will downsample these.
TIER_3_SOURCES = {
    "olympiads"       # ~117k
}

print("Filtering and Balancing Candidates...")

candidates = []
tier3_candidates = []

for row in tqdm(ds):
    # Basic Filter: Must be a word problem
    if row.get("question_type") != "math-word-problem":
        continue
        
    src = row.get("source")
    
    if src in TIER_1_SOURCES or src in TIER_2_SOURCES:
        candidates.append(row)
    elif src in TIER_3_SOURCES:
        tier3_candidates.append(row)

# 2. Downsample Tier 3 (Olympiads)
# We have ~20k from Tiers 1&2. We need ~30k more to reach a safe buffer of 50-60k.
target_tier3 = 40_000
if len(tier3_candidates) > target_tier3:
    print(f"Downsampling 'olympiads' from {len(tier3_candidates)} to {target_tier3}...")
    tier3_candidates = random.sample(tier3_candidates, target_tier3)

# 3. Combine and Shuffle
candidates.extend(tier3_candidates)
random.shuffle(candidates) # CRITICAL: Mix them so we generate variety in the first 4 hours

print(f"Final Candidate Count: {len(candidates)}")
print(f"  - Tier 1&2 (High Priority): ~20,000")
print(f"  - Tier 3 (Olympiads): {len(tier3_candidates)}")

if len(candidates) == 0:
    print("CRITICAL ERROR: No candidates found.")
    sys.exit(1)

# --- MODEL INFERENCE (Standard vLLM setup) ---
print("Initializing vLLM...")
llm = LLM(
    model="Qwen/Qwen2.5-Math-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90, 
    max_model_len=4096,
    trust_remote_code=True
)

params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)

# --- BATCH GENERATION ---
written = 0

for i in range(0, len(candidates), BATCH_SIZE):
    if written >= TARGET_COUNT:
        break
        
    batch = candidates[i : i + BATCH_SIZE]
    prompts = [row["problem"] for row in batch]
    
    print(f"Generating batch {i} - {i+len(batch)}...")
    outputs = llm.generate(prompts, params)
    
    new_samples = []
    
    for row, output in zip(batch, outputs):
        wrong_sol = output.outputs[0].text.strip()
        correct_ans = str(row["answer"]).strip()
        
        # Robust check: Answer shouldn't be at the end of the wrong solution
        if correct_ans in wrong_sol[-200:]: 
            continue 

        # Construct Recursive Sample
        full_content = (
            f"<think>{wrong_sol}</think>\n"
            f"<wait>\n"
            f"<critique>Wait, I made a mistake. Let me re-calculate.</critique>\n"
            f"<think>{row['solution']}</think>\n"
            f"<answer>{row['answer']}</answer>"
        )

        sample = {
            "source": "recursive_correction",
            "problem": row["problem"],
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": full_content}
            ]
        }
        new_samples.append(ujson.dumps(sample))
    
    if new_samples:
        with OUT.open("a") as f:
            for line in new_samples:
                f.write(line + "\n")
            
    written += len(new_samples)
    print(f"Progress: {written}/{TARGET_COUNT} samples saved.")

print("Done.")
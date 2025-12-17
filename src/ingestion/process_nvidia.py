import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# Config
OUT_PATH = Path("data/processed/code_silver.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 200_000 # Configurable

def process_nvidia():
    print(f"Streaming OpenMathInstruct-2 (Target: {MAX_SAMPLES})...")
    
    ds = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train_5M",
        streaming=True
    )

    written = 0
    
    with open(OUT_PATH, "w") as f:
        for row in tqdm(ds):
            # 1. Filter Source
            src = row.get("problem_source", "unknown")
            if src not in ["math", "augmented_math"]:
                continue
                
            # 2. Filter Length
            if len(row["problem"]) > 4000:
                continue

            # 3. Filter Integer Answer
            try:
                ans_str = row.get("expected_answer", "0")
                if "." in ans_str:
                    if not float(ans_str).is_integer():
                        continue
            except:
                continue

            # 4. Universal Schema (Wrap in <think>)
            sample = {
                "id": f"nvidia_om2_{written}",
                "source": "nvidia_openmath_2",
                "problem": row["problem"],
                "messages": [
                    {"role": "user", "content": row["problem"]},
                    {
                        "role": "assistant",
                        "content": f"<think>{row['generated_solution']}</think>\n<answer>{row['expected_answer']}</answer>"
                    }
                ],
                "meta": {
                    "source_type": src,
                    "is_code": True
                }
            }
            
            f.write(ujson.dumps(sample) + "\n")
            written += 1
            
            if written >= MAX_SAMPLES:
                break
    
    print(f"Successfully wrote {written} samples to {OUT_PATH}")

if __name__ == "__main__":
    process_nvidia()
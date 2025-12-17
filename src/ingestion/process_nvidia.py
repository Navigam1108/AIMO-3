import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# Config
OUT_PATH = Path("data/processed/code_silver.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 2_000_000

def process_nvidia():
    print("Streaming OpenMathInstruct-2 (train_5M)...")
    
    # Load the 5M subset to save RAM/Bandwidth
    ds = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train_5M",
        streaming=True
    )

    written = 0
    
    with open(OUT_PATH, "w") as f:
        for row in tqdm(ds):
            # 1. Fix: Filter for lowercase 'math' and 'augmented_math'
            # (GSM8K is too easy for AIMO, so we skip it)
            src = row.get("problem_source", "unknown")
            if src not in ["math", "augmented_math"]:
                continue
                
            # 2. Fix: Check length (NVIDIA recommends dropping > 1024 tokens)
            if len(row["problem"]) > 4000:
                continue

            # 3. Fix: Ensure integer answers (AIMO constraint)
            try:
                # 'expected_answer' is the correct key
                ans_str = row.get("expected_answer", "0")
                if "." in ans_str:
                    ans_val = float(ans_str)
                    if not ans_val.is_integer():
                        continue
            except:
                continue

            # 4. Fix: Use correct keys 'generated_solution' and 'expected_answer'
            # Note: The solution often contains mixed text/code.
            # We wrap it in <think> as a baseline, but ideally, you'd parse it.
            # For Silver data, simple wrapping is acceptable.
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
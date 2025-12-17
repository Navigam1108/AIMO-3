import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

OUT = Path("data/processed/logic_core.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading NuminaMath-1.5...")
ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

def valid(row):
    # 1. Type Filter
    if row.get("question_type") != "math-word-problem":
        return False
    
    # 2. Source Filter (No synthetic)
    src = row.get("source", "")
    if "synthetic_math" in src or "synthetic_amc" in src:
        return False
    
    # 3. Answer Filter
    ans = row.get("answer")
    if ans is None:
        return False
    try:
        # Check if it looks like a number (allows "100" or "100.0" but filters text)
        float(ans) 
    except:
        return False
    return True

with OUT.open("w") as f:
    for row in tqdm(ds):
        if not valid(row):
            continue

        sample = {
            "source": "numina_math_1.5",
            "problem": row["problem"],
            "messages": [
                {"role": "user", "content": row["problem"]},
                {
                    "role": "assistant",
                    # Wrap solution in <think>
                    "content": f"<think>{row['solution']}</think>\n<answer>{row['answer']}</answer>"
                }
            ]
        }
        f.write(ujson.dumps(sample) + "\n")
import json
import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

OUT = Path("data/processed/logic_core.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

def valid(row):
    if row.get("question_type") != "math-word-problem":
        return False
    src = row.get("source", "")
    if "synthetic_math" in src or "synthetic_amc" in src:
        return False
    try:
        int(row["answer"])
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
                    "content": f"<think>{row['solution']}</think>\n<answer>{row['answer']}</answer>"
                }
            ]
        }
        f.write(ujson.dumps(sample) + "\n")

import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import re

OUT = Path("data/processed/code_plat.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")

def has_valid_python(code):
    return "```python" in code

with OUT.open("w") as f:
    for row in tqdm(ds):
        sol = row["solution"]
        if not has_valid_python(sol):
            continue

        sample = {
            "source": "numina_tir",
            "problem": row["problem"],
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": sol}
            ]
        }
        f.write(ujson.dumps(sample) + "\n")

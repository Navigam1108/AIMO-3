import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

OUT = Path("data/processed/code_silver.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train_5M",
    streaming=True
)

MAX_SAMPLES = 200_000
written = 0

with OUT.open("w") as f:
    for row in tqdm(ds):
        if row.get("problem_source") != "MATH":
            continue
        if len(row["problem"]) > 4000:
            continue

        sample = {
            "source": "nvidia_openmath",
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
        written += 1
        if written >= MAX_SAMPLES:
            break

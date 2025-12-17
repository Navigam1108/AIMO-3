import ujson
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

OUT = Path("data/processed/code_plat.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading NuminaMath-TIR...")
ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")

def has_valid_python(code):
    return "```python" in code

with OUT.open("w") as f:
    for row in tqdm(ds):
        sol = row["solution"]
        # Basic quality check
        if not has_valid_python(sol):
            continue

        # --- SCHEMA FIX ---
        # Wrap the code-integrated solution in <think> tags.
        # The Numina TIR dataset usually ends with the final answer inside the text,
        # but to be safe and uniform, we wrap the whole reasoning/coding block.
        content_body = f"<think>{sol}</think>"
        
        # If the dataset has a separate 'answer' field (it usually does), append it.
        # NuminaTIR rows usually have 'solution' and 'answer'.
        if "answer" in row and row["answer"]:
             content_body += f"\n<answer>{row['answer']}</answer>"

        sample = {
            "source": "numina_tir",
            "problem": row["problem"],
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": content_body}
            ]
        }
        f.write(ujson.dumps(sample) + "\n")
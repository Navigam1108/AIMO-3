import ujson
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path

OUT = Path("data/processed/recursive.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

llm = LLM(
    model="Qwen/Qwen2.5-Math-7B-Instruct",
    tensor_parallel_size=1
)

params = SamplingParams(temperature=1.0, top_p=0.95)

base_ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

def gt_answer(row):
    return str(row["answer"]).strip()

written = 0
TARGET = 50_000

with OUT.open("w") as f:
    for row in tqdm(base_ds):
        if written >= TARGET:
            break

        prompt = row["problem"]
        outputs = llm.generate([prompt], params)
        wrong = outputs[0].outputs[0].text.strip()

        if gt_answer(row) in wrong:
            continue

        sample = {
            "source": "recursive_correction",
            "problem": prompt,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": wrong},
                {"role": "assistant", "content": "<critique>Wait, I made a mistake. Let me re-calculate.</critique>"},
                {
                    "role": "assistant",
                    "content": f"<think>{row['solution']}</think>\n<answer>{row['answer']}</answer>"
                }
            ]
        }
        f.write(ujson.dumps(sample) + "\n")
        written += 1

import random
import ujson
from pathlib import Path
from tqdm import tqdm

FILES = {
    "logic_core": ("data/processed/logic_core.jsonl", 1.0),
    "code_plat": ("data/processed/code_plat.jsonl", 3.0),
    "code_silver": ("data/processed/code_silver.jsonl", 0.5),
    "recursive": ("data/processed/recursive.jsonl", 1.0),
}

OUT = Path("data/gold/aimo_system2_final.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

buffers = []

for name, (path, weight) in FILES.items():
    with open(path) as f:
        rows = [ujson.loads(l) for l in f]
        k = int(len(rows) * weight)
        buffers.extend(random.choices(rows, k=k))

random.shuffle(buffers)

with OUT.open("w") as f:
    for r in tqdm(buffers):
        f.write(ujson.dumps(r) + "\n")

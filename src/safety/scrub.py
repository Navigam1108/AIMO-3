from datasketch import MinHash, MinHashLSH
from pathlib import Path
import ujson
from tqdm import tqdm

BLOCKLIST = Path("data/blocklist/aimo_ref.txt")
DATA_DIR = Path("data/processed")
THRESHOLD = 0.85

def mh(text):
    m = MinHash(num_perm=128)
    for w in text.split():
        m.update(w.encode("utf8"))
    return m

# Load blocklist
block_hashes = {}
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)

with BLOCKLIST.open() as f:
    for i, line in enumerate(f):
        m = mh(line.strip())
        lsh.insert(f"ref_{i}", m)
        block_hashes[f"ref_{i}"] = m

def scrub_file(path):
    cleaned = []
    removed = 0

    with path.open() as f:
        for line in f:
            row = ujson.loads(line)
            m = mh(row["problem"])
            if lsh.query(m):
                removed += 1
                continue
            cleaned.append(row)

    with path.open("w") as f:
        for r in cleaned:
            f.write(ujson.dumps(r) + "\n")

    return removed

total_removed = 0
for file in DATA_DIR.glob("*.jsonl"):
    total_removed += scrub_file(file)

print(f"Total removed: {total_removed}")

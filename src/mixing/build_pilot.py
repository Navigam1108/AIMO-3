import ujson
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = Path("/teamspace/studios/this_studio/aimo_datafoundry/data/processed")
OUTPUT_FILE = Path("/teamspace/studios/this_studio/aimo_datafoundry/data/gold/pilot_micro.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Target: ~25k - 30k total
SAMPLING_PLAN = {
    # 70k total -> Take 20% (~14k). High quality code is our priority.
    "code_plat.jsonl": 0.20,      
    
    # 150k total -> Take 5% (~7.5k). Just enough to keep reasoning intact.
    "logic_core.jsonl": 0.05,    
    
    # 200k total -> Take 2.5% (~5k). Just a flavor of synthetic data.
    "code_silver.jsonl": 0.025,   
}

def build_micro():
    print("🔬 Building Micro-Pilot Dataset (~30k)...")
    final_data = []
    
    for filename, percent in SAMPLING_PLAN.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"⚠️ MISSING: {filename}")
            continue
            
        with filepath.open("r") as f:
            rows = [ujson.loads(line) for line in f]
            
        count = int(len(rows) * percent)
        print(f"  - {filename}: {len(rows)} rows -> Taking {count}")
        
        sampled = random.sample(rows, count)
        final_data.extend(sampled)

    random.shuffle(final_data)
    
    print(f"💾 Saving {len(final_data)} samples to {OUTPUT_FILE}...")
    with OUTPUT_FILE.open("w") as f:
        for row in tqdm(final_data):
            f.write(ujson.dumps(row) + "\n")

if __name__ == "__main__":
    build_micro()
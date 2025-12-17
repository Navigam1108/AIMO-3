import ujson
from pathlib import Path
from tqdm import tqdm
import shutil

DATA_DIR = Path("data/processed")

def add_ids_to_file(path):
    print(f"Processing {path.name}...")
    temp_path = path.with_suffix(".tmp")
    
    # Use the filename (without extension) as the ID prefix
    # e.g., code_silver.jsonl -> "code_silver_0", "code_silver_1"...
    prefix = path.stem 
    
    count = 0
    with path.open("r") as fin, temp_path.open("w") as fout:
        for i, line in enumerate(fin):
            try:
                row = ujson.loads(line)
            except ValueError:
                continue # Skip broken lines

            # Add ID if missing
            if "id" not in row:
                row["id"] = f"{prefix}_{i}"
            
            fout.write(ujson.dumps(row) + "\n")
            count += 1
            
    # Safely replace the old file
    shutil.move(temp_path, path)
    print(f"-> Updated {count} rows in {path.name}")

if __name__ == "__main__":
    for file in DATA_DIR.glob("*.jsonl"):
        add_ids_to_file(file)
import ujson
import os
import shutil
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to the text file containing the 10 banned problems
BLOCKLIST_PATH = Path("data/blocklist/aimo_ref.txt")

# Directory containing your .jsonl files to clean
DATA_DIR = Path("data/processed")

# Similarity threshold (0.85 = Very strict, 0.5 = Loose)
# 0.85 means "If 85% of the 3-word phrases match, delete it."
THRESHOLD = 0.85 

# Num Permutations for MinHash (128 is standard tradeoff for speed/accuracy)
NUM_PERM = 128

def get_minhash(text):
    """
    Converts text into a MinHash signature using 3-gram shingling.
    This makes matching robust to small edits or formatting changes.
    """
    m = MinHash(num_perm=NUM_PERM)
    # Normalize: lowercase and simple split
    tokens = text.lower().split()
    
    # Create 3-grams (sliding window of 3 words)
    # e.g., "alice and bob" -> "alice and bob"
    if len(tokens) < 3:
        # Fallback for very short lines
        m.update(" ".join(tokens).encode("utf8"))
    else:
        for i in range(len(tokens) - 2):
            shingle = " ".join(tokens[i:i+3])
            m.update(shingle.encode("utf8"))
    return m

def load_blocklist():
    """
    Loads the blocklist and builds the LSH Index.
    Treats EVERY non-empty line as a banned signature.
    """
    print(f"🔒 Loading Blocklist from {BLOCKLIST_PATH}...")
    
    if not BLOCKLIST_PATH.exists():
        raise FileNotFoundError(f"CRITICAL: Blocklist not found at {BLOCKLIST_PATH}. Please create it!")

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    
    with BLOCKLIST_PATH.open("r", encoding="utf-8") as f:
        # Read lines, strip whitespace, remove empty lines
        lines = [line.strip() for line in f if line.strip()]
    
    count = 0
    for i, line in enumerate(lines):
        # Skip lines that are too short to be unique (e.g., "Problem 1")
        if len(line.split()) < 4:
            continue
            
        m = get_minhash(line)
        # Use line index as key
        lsh.insert(f"ref_{i}", m)
        count += 1
        
    print(f"✅ Indexed {count} unique signatures from the blocklist.")
    return lsh

def scrub_files():
    # 1. Build the Safety Net
    lsh = load_blocklist()
    
    total_removed = 0
    files = list(DATA_DIR.glob("*.jsonl"))
    
    print(f"\n🧹 Starting Scrub on {len(files)} files...")

    for file_path in files:
        print(f"Processing {file_path.name}...")
        
        # We write to a temp file to avoid corrupting data on crash
        temp_path = file_path.with_suffix(".tmp")
        removed_in_file = 0
        kept_in_file = 0
        
        with file_path.open("r", encoding="utf-8") as fin, \
             temp_path.open("w", encoding="utf-8") as fout:
            
            for line in tqdm(fin, desc=f"Scanning {file_path.name}", unit="rows"):
                try:
                    row = ujson.loads(line)
                except ValueError:
                    continue # Skip broken JSON lines
                
                # Check 1: Content Similarity
                # We check the 'problem' text against the blocklist
                prob_text = row.get("problem", "")
                if not prob_text:
                    continue
                    
                m = get_minhash(prob_text)
                
                # Query LSH
                matches = lsh.query(m)
                
                if len(matches) > 0:
                    removed_in_file += 1
                    # Optional: Log the hit
                    print(f"  [HIT] Removed ID {row.get('id')} (Matches: {matches})")
                    continue
                
                # Check 2: Hard Keyword Check (Safety Belt)
                # Sometimes LSH misses if the overlap is small but specific.
                # Banning specific recent years is a good heuristic.
                keywords = ["AIME 2024", "AIME 2025", "AIMO 2024", "AIMO 2025"]
                if any(k in prob_text for k in keywords):
                     removed_in_file += 1
                     continue

                # If safe, write to temp file
                fout.write(line)
                kept_in_file += 1

        # Safety Atomic Swap
        # Only replace the original file if the write finished successfully
        shutil.move(temp_path, file_path)
        
        print(f"  -> Removed: {removed_in_file} | Kept: {kept_in_file}")
        total_removed += removed_in_file

    print(f"\n🎉 Scrub Complete. Total Contaminated Samples Removed: {total_removed}")

if __name__ == "__main__":
    scrub_files()
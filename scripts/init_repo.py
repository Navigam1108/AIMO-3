from pathlib import Path

ROOT = Path("aimo_datafoundry")

STRUCTURE = {
    "data": {
        "raw": {},
        "processed": {},
        "blocklist": {
            "aimo_ref.txt": ""
        },
        "gold": {}
    },
    "src": {
        "ingestion": {
            "process_numina.py": "",
            "process_tir.py": "",
            "process_nvidia.py": ""
        },
        "safety": {
            "scrub.py": ""
        },
        "synthesis": {
            "generate_recursive.py": ""
        },
        "mixing": {
            "build_mix.py": ""
        }
    },
    "requirements.txt": ""
}

def create_tree(base: Path, tree: dict):
    for name, content in tree.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_tree(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            if content:
                path.write_text(content)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    create_tree(ROOT, STRUCTURE)
    print(f"Repository initialized at: {ROOT.resolve()}")

if __name__ == "__main__":
    main()

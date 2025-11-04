"""
Download TCGA WSI slides 
"""

import os, json, requests, argparse
from pathlib import Path

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def download_one(file_id: str, out_path: Path, timeout=600):
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    ensure_dir(out_path.parent)
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return
    print(f"[dl] {out_path.name}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_from_index(index_path: Path, base_dir: Path):
    with index_path.open() as f:
        for line in f:
            rec = json.loads(line)
            file_id = rec["file_id"]
            local_path = base_dir / rec["local_path"]
            download_one(file_id, local_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()

    data_root = Path("data/index")
    splits= []
    if args.split in ("all","train"):
        splits.append(data_root / "train/index.jsonl")
    if args.split in ("all","test"):
        splits.append(data_root / "test/index.jsonl")

    for idx in splits:
        print(f"Processing {idx}")
        download_from_index(idx, Path("."))

if __name__ == "__main__":
    main()

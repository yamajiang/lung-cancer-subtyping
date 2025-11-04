"""
Build index for LUAD vs LUSC WSI classification
Fetches TCGA slides via the GDC API, stores metadata, and splits into train/val/test 
75/15/10 split
"""
import os
import json 
import argparse
import random
import requests
from pathlib import Path

FILES= "https://api.gdc.cancer.gov/files"
DATA= "https://api.gdc.cancer.gov/data"

def ensure_dir(p:Path) -> None:
    p.mkdir(parents=True, exist_ok= True)

def gdc_query(project_id: str, page_size=1000):
    filters= {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }
    payload= {"filters": filters, "fields": "file_id,file_name,cases.case_id,cases.submitter_id", "size": page_size}
    r= requests.post(FILES, json=payload)
    r.raise_for_status()
    return r.json()["data"]["hits"]

def build_records(project_id, label):
    slides= gdc_query(project_id)
    recs= []
    for s in slides:
        case= s["cases"][0]
        recs.append({
            "case_id": case["case_id"],
            "case_submitter_id": case["submitter_id"],
            "project_id": project_id,
            "label": label,
            "file_id": s["file_id"],
            "file_name": s["file_name"],
            "download_url": f"{DATA}/{s['file_id']}",
            "local_path": f"data/raw/{project_id}/slides/{s['file_name']}"
        })
    return recs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path("data/index")
    ensure_dir(data_root / "train")
    ensure_dir(data_root / "test")

    luad = build_records("TCGA-LUAD", "LUAD")
    lusc = build_records("TCGA-LUSC", "LUSC")
    full = luad + lusc
    print(f"Total slides: {len(full)} (LUAD={len(luad)}, LUSC={len(lusc)})")

    with (data_root / "index_full.jsonl").open("w") as f:
        for r in full:
            f.write(json.dumps(r) + "\n")

    #split by case
    random.seed(args.seed)
    cases = list({r["case_id"] for r in full})
    random.shuffle(cases)

    n_train = int(len(cases) * 0.75)
    n_val = int(len(cases) * 0.15)
    # remaining goes to test
    train_cases = set(cases[:n_train])
    val_cases = set(cases[n_train:n_train + n_val])
    test_cases = set(cases[n_train + n_val:])

    train= [r for r in full if r["case_id"] in train_cases]
    val= [r for r in full if r["case_id"] in val_cases]
    test= [r for r in full if r["case_id"] in test_cases]

    # save split indices
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        folder= data_root / name
        ensure_dir(folder)
        with (folder / "index.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {name}/index.jsonl with {len(recs)} samples")

if __name__ == "__main__":
    main()
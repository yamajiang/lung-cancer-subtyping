"""
Download GDC clinical and biospecimen metadata in json
for TCGA-LUAD and TCGA-LUSC saved under data/metadata/
"""

import requests
from pathlib import Path
import sys

PROJECTS= ["TCGA-LUAD", "TCGA-LUSC"]
DATA_CATEGORIES= ["Clinical", "Biospecimen"]
FILES= "https://api.gdc.cancer.gov/files"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fetch_metadata(project_id: str, data_category: str):
    filters={
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_category", "value": [data_category]}},
        ],
    }

    payload= {
        "filters": filters,
        "format": "JSON", 
        "size": "10000",
    }

    print(f"Fetching {data_category} metadata for {project_id} as JSON...")
    r= requests.post(FILES, json=payload)
    r.raise_for_status()
    return r.content

def main():
    out_dir = Path("data/metadata")
    ensure_dir(out_dir)

    for proj in PROJECTS:
        for cat in DATA_CATEGORIES:
            try:
                content= fetch_metadata(proj, cat)
                outfile= out_dir / f"{proj}_{cat.lower()}.json"
                with open(outfile, "wb") as f:
                    f.write(content)
                print(f"Saved {outfile}")
            except Exception as e:
                print(f"Failed for {proj} {cat}: {e}", file=sys.stderr)

    print("\nAll metadata saved to data/metadata/")

if __name__ == "__main__":
    main()

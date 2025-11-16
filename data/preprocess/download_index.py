"""
Process TCGA WSI slides one-by-one to save space:
- download -> tile -> embed with Prov-GigaPath -> save .pt -> cleanup

Index format: JSONL with at least {"file_id": "...", "local_path": "data/raw/{project_id}/slides/{file_name}.svs"}

Embeddings are saved to --embeds_dir as <slide_stem>.pt

Requirements:
  pip install prov-gigapath timm torch torchvision pillow requests tqdm
  export HF_TOKEN=...  # if your env needs HF auth for model fetch
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

import torch
import timm
from gigapath import pipeline
import gigapath  # for slide encoder factory

# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def replace_suffix(path: Path, new_suffix: str) -> Path:
    return path.with_suffix(new_suffix)

def download_one(file_id: str, out_path: Path, timeout: int = 600, gdc_token: str | None = None):
    """
    Stream a file by UUID from the GDC data endpoint to out_path.
    """
    ensure_dir(out_path.parent)
    if out_path.exists():
        return
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    headers = {}
    if gdc_token:
        headers["X-Auth-Token"] = gdc_token
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def read_index_lines(index_path: Path):
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# ----------------------------
# GigaPath model loading
# ----------------------------
def load_gigapath_models(tile_encoder_id: str, slide_encoder_arch: str, slide_d_model: int, device: str):
    # Tile encoder from HF Hub via timm, as in Prov-GigaPath README
    tile_encoder = timm.create_model(tile_encoder_id, pretrained=True)
    tile_encoder.eval()

    # Slide encoder from gigapath factory, as in README
    slide_encoder = gigapath.slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath", slide_encoder_arch, slide_d_model
    )
    slide_encoder.eval()

    # Move to device if available
    if device == "cuda" and torch.cuda.is_available():
        tile_encoder = tile_encoder.to("cuda")
        slide_encoder = slide_encoder.to("cuda")

    return tile_encoder, slide_encoder

# ----------------------------
# One-slide pipeline
# ----------------------------
def process_one_slide(
    file_id: str,
    slide_path: Path,
    tmp_root: Path,
    embeds_dir: Path,
    tile_level: int,
    tile_encoder,
    slide_encoder,
    timeout: int,
    gdc_token: str | None,
):
    """
    If <embeds_dir>/<slide_stem>.pt exists -> skip.
    Else:
      download .svs -> tile -> tile+slide encoders -> save .pt -> cleanup tiles and .svs
    """
    ensure_dir(embeds_dir)
    embed_path = embeds_dir / (slide_path.stem + ".pt")
    if embed_path.exists():
        return "skipped", embed_path

    # A per-slide working directory so an interrupted run is easy to clean
    work_dir = tmp_root / slide_path.stem
    tiles_parent = work_dir  # gigapath pipeline writes output/<basename> under save_dir
    ensure_dir(work_dir)

    try:
        # 1) Download the WSI
        download_one(file_id, slide_path, timeout=timeout, gdc_token=gdc_token)

        # 2) Tile
        pipeline.tile_one_slide(str(slide_path), save_dir=str(work_dir), level=tile_level)

        # Gigapath writes tiles to: <save_dir>/output/<basename>/
        slide_tiles_dir = tiles_parent / "output" / slide_path.name
        if not slide_tiles_dir.exists():
            raise FileNotFoundError(f"Tiles directory not found: {slide_tiles_dir}")

        image_paths = [
            str(slide_tiles_dir / img)
            for img in os.listdir(slide_tiles_dir)
            if img.endswith(".png")
        ]
        if not image_paths:
            raise RuntimeError(f"No PNG tiles found in {slide_tiles_dir}")

        # 3) Tile encoder then slide encoder
        tile_encoder_outputs = pipeline.run_inference_with_tile_encoder(image_paths, tile_encoder, batch_size=128)
        slide_embeds = pipeline.run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder, **tile_encoder_outputs
        )

        # 4) Save only the slide embedding tensor/object
        torch.save(slide_embeds, embed_path)

        return "ok", embed_path
    finally:
        # 5) Cleanup tiles and the large slide to save space
        if slide_path.exists():
            try:
                slide_path.unlink()
            except Exception:
                pass
        if tiles_parent.exists():
            try:
                shutil.rmtree(tiles_parent)
            except Exception:
                pass

# ----------------------------
# Main
# ----------------------------
def gather_index_files(index_root: Path, split: str):
    splits = []
    if split in ("all", "train"):
        splits.append(index_root / "train" / "index.jsonl")
    if split in ("all", "test"):
        splits.append(index_root / "test" / "index.jsonl")
    return [p for p in splits if p.exists()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_root", type=str, default="data/index")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--tmp_dir", type=str, default="data/tmp")
    parser.add_argument("--embeds_dir", type=str, default="data/embeddings")
    parser.add_argument("--tile_level", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gdc_token", type=str, default="")  # optional for controlled data

    # Model knobs with sensible defaults from the Prov-GigaPath README
    parser.add_argument("--tile_encoder_id", type=str, default="hf_hub:prov-gigapath/prov-gigapath")
    parser.add_argument("--slide_encoder_arch", type=str, default="gigapath_slide_enc12l768d")
    parser.add_argument("--slide_d_model", type=int, default=1536)

    args = parser.parse_args()

    index_root = Path(args.index_root)
    base_dir = Path(args.base_dir)
    tmp_root = Path(args.tmp_dir)
    embeds_dir = Path(args.embeds_dir)
    ensure_dir(tmp_root)
    ensure_dir(embeds_dir)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tile_encoder, slide_encoder = load_gigapath_models(
        args.tile_encoder_id, args.slide_encoder_arch, args.slide_d_model, device
    )

    index_files = gather_index_files(index_root, args.split)
    if not index_files:
        raise FileNotFoundError(f"No index files found under {index_root} for split={args.split}")

    # Build the full record list so tqdm can show one progress bar across all files
    records = []
    for idx_path in index_files:
        records.extend(list(read_index_lines(idx_path)))

    pbar = tqdm(records, desc="Slides", unit="slide")
    for rec in pbar:
        file_id = rec["file_id"]
        slide_rel = Path(rec["local_path"])
        slide_path = base_dir / slide_rel
        embed_path = embeds_dir / (slide_path.stem + ".pt")

        if embed_path.exists():
            pbar.set_postfix_str(f"skip {slide_path.stem}")
            continue

        pbar.set_postfix_str(f"proc {slide_path.stem}")
        status, out_path = process_one_slide(
            file_id=file_id,
            slide_path=slide_path,
            tmp_root=tmp_root,
            embeds_dir=embeds_dir,
            tile_level=args.tile_level,
            tile_encoder=tile_encoder,
            slide_encoder=slide_encoder,
            timeout=args.timeout,
            gdc_token=(args.gdc_token or None),
        )
        if status != "ok":
            # Let it raise if something went wrong next time; minimal handling per request
            pass

if __name__ == "__main__":
    main()

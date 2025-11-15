from pathlib import Path 
import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide
from tqdm import tqdm
from gigapath.pipeline import tile_one_slide 
from __future__ import annotations
from functools import partial 
from pathlib import Path

#tile slide using gigapath and skip the background 
def title_single_slide(slide_path: Path, output_dir:Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    done_flag = output_dir / "done.flag"
    if done_flag.exists():
        return {"slide": slide_path.name, "tiles_dir": str(output_dir)}

    tile_one_slide(wsi_path=str(slide_path), save_dir=str(output_dir), level=1, tissue_seg=True, skip_background=True)
    return {"slide": slide_path.name, "tiles_dir": str(output_dir)}
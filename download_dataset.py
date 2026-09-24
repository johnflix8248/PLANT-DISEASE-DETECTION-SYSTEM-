"""
Download the iBean plant-disease dataset (Makerere AI Lab).

Three classes:
  - angular_leaf_spot  (diseased)
  - bean_rust          (diseased)
  - healthy

Source (public, no login):
  https://github.com/AI-Lab-Makerere/ibean
  https://storage.googleapis.com/ibean/
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_ROOT = Path(__file__).resolve().parent / "dataset"
SPLITS = {
    "train": "https://storage.googleapis.com/ibean/train.zip",
    "validation": "https://storage.googleapis.com/ibean/validation.zip",
    "test": "https://storage.googleapis.com/ibean/test.zip",
}


def _download(url: str, dest: Path, attempts: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            urlretrieve(url, dest)
            return
        except Exception as exc:  # network / SSL glitches
            last_err = exc
            print(f"  attempt {i}/{attempts} failed: {exc}")
    raise RuntimeError(f"Could not download {url}") from last_err


def download_dataset(force: bool = False) -> Path:
    """
    Download and extract train/validation/test zips into ./dataset.
    Returns the dataset root path.
    """
    marker = DATA_ROOT / ".ready"
    if marker.exists() and not force:
        print(f"Dataset already present at {DATA_ROOT}")
        return DATA_ROOT

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    zips_dir = DATA_ROOT / "_zips"
    zips_dir.mkdir(exist_ok=True)

    for split, url in SPLITS.items():
        zip_path = zips_dir / f"{split}.zip"
        if force or not zip_path.exists():
            _download(url, zip_path)
        extract_to = DATA_ROOT / split
        if extract_to.exists() and force:
            shutil.rmtree(extract_to)
        if not extract_to.exists():
            print(f"Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_ROOT)
            # Some zips unpack as train/train — flatten if needed
            nested = DATA_ROOT / split / split
            if nested.is_dir():
                for child in nested.iterdir():
                    shutil.move(str(child), str(extract_to / child.name))
                shutil.rmtree(nested)

    marker.write_text("ibean plant disease dataset\n")
    print(f"Dataset ready at {DATA_ROOT}")
    _print_summary()
    return DATA_ROOT


def _print_summary() -> None:
    print("\nClass counts:")
    for split in ("train", "validation", "test"):
        split_dir = DATA_ROOT / split
        if not split_dir.is_dir():
            continue
        print(f"  [{split}]")
        for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(1 for _ in cls.rglob("*") if _.is_file())
            print(f"    {cls.name}: {n} images")


if __name__ == "__main__":
    download_dataset()

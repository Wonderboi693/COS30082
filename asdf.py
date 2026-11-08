import os
import shutil
from pathlib import Path
from tqdm import tqdm

from src.config import Config

PROJECT_ROOT = Path.cwd().resolve()
TARGET_DIR = Path(Config.EMOTION_DIR)

# RAF-DB 7 basic emotions mapping (1-indexed)
RAF_LABEL_MAP = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Angry",
    7: "Neutral"
}

def setup_target_dirs():
    for split in ["train", "val"]:
        for cls_name in Config.EMOTION_CLASSES:
            (TARGET_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

def organize_from_annotation_file(raf_root: Path, annotation_file: Path):
    """
    Handles official RAF-DB distribution:
    list_patition_label.txt format: '<filename_aligned.jpg> <label_int>'
    """
    print(f"[*] Found annotation list: {annotation_file.name}")
    # Locate images folder (can be aligned/, Image/aligned/, or flat)
    possible_img_dirs = [
        raf_root / "DATASET" / "Image" / "aligned",
        raf_root / "Image" / "aligned",
        raf_root / "aligned",
        raf_root
    ]
    img_dir = next((d for d in possible_img_dirs if d.is_dir()), raf_root)
    print(f"[*] Reading image crops from: {img_dir}")

    counts = {"train": 0, "val": 0}

    with open(annotation_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc="Organizing RAF-DB"):
        parts = line.split()
        if len(parts) < 2:
            continue
        
        file_name, label_id = parts[0], int(parts[1])
        cls_name = RAF_LABEL_MAP.get(label_id)
        if not cls_name:
            continue

        split = "train" if file_name.startswith("train_") else "val"

        # Search for file variants (e.g., train_00001_aligned.jpg)
        src_file = img_dir / file_name
        if not src_file.exists():
            base_stem = file_name.split(".")[0]
            aligned_candidate = img_dir / f"{base_stem}_aligned.jpg"
            if aligned_candidate.exists():
                src_file = aligned_candidate
            else:
                continue

        dest_file = TARGET_DIR / split / cls_name / src_file.name
        shutil.copy2(src_file, dest_file)
        counts[split] += 1

    return counts

def organize_from_prebuilt_folders(raf_root: Path):
    """
    Handles Kaggle mirrors with pre-sorted folders:
    e.g. raf-db/train/1/.. or raf-db/train/happy/..
    """
    print("[*] Parsing folder structure directly...")
    counts = {"train": 0, "val": 0}

    # Find where train/ and test/ or val/ reside
    subdirs = {d.name.lower(): d for d in raf_root.rglob("*") if d.is_dir()}
    train_root = subdirs.get("train")
    test_root = subdirs.get("test") or subdirs.get("val")

    if not train_root or not test_root:
        raise FileNotFoundError(f"Could not locate train and test/val folders under {raf_root}")

    for src_split_dir, target_split in [(train_root, "train"), (test_root, "val")]:
        for folder in src_split_dir.iterdir():
            if not folder.is_dir():
                continue

            # Determine class: could be digit '1'-'7' or text name
            folder_name = folder.name
            if folder_name.isdigit():
                cls_name = RAF_LABEL_MAP.get(int(folder_name))
            else:
                cls_match = [c for c in Config.EMOTION_CLASSES if c.lower() in folder_name.lower()]
                cls_name = cls_match[0] if cls_match else None

            if not cls_name:
                continue

            for img_file in folder.glob("*.*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    dest = TARGET_DIR / target_split / cls_name / img_file.name
                    shutil.copy2(img_file, dest)
                    counts[target_split] += 1

    return counts

def main():
    setup_target_dirs()

    # Search for RAF-DB root inside data/raw
    potential_roots = [
        PROJECT_ROOT / "data" / "raw" / "RAF-DB",
        PROJECT_ROOT / "data" / "raw" / "raf-db",
        PROJECT_ROOT / "data" / "raw" / "raf_db"
    ]
    raf_root = next((p for p in potential_roots if p.is_dir()), None)

    if not raf_root:
        print("[!] RAF-DB directory not found in data/raw/. Checked:")
        for p in potential_roots:
            print(f"    - {p}")
        print("\nPlace the RAF-DB archive in data/raw/RAF-DB and run again.")
        return

    # Check for annotation list (spelled 'list_patition_label.txt' in official release)
    ann_files = list(raf_root.rglob("*patition_label.txt")) + list(raf_root.rglob("*partition_label.txt"))

    if ann_files:
        counts = organize_from_annotation_file(raf_root, ann_files[0])
    else:
        counts = organize_from_prebuilt_folders(raf_root)

    print("\n" + "=" * 45)
    print("  RAF-DB PREPARATION COMPLETE")
    print("=" * 45)
    print(f"  Training Samples:   {counts['train']:>6}")
    print(f"  Validation Samples: {counts['val']:>6}")
    print("=" * 45)
    print(f"[✓] Formatted dataset ready at: {TARGET_DIR}")

if __name__ == "__main__":
    main()
import json
import math
from pathlib import Path

DATASET_ROOT = Path("dataset/train")
OUT_PATH = DATASET_ROOT / "metadata.jsonl"

MIN_REPEATS = 1
MAX_REPEATS = 4  # safety cap to avoid extreme oversampling for tiny categories
IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")


def read_caption(img_path: Path) -> str:
    txt_path = img_path.with_suffix(".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing caption: {txt_path}")
    return txt_path.read_text(encoding="utf-8").strip()


def list_images(cat_dir: Path):
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(cat_dir.glob(f"*.{ext}"))
        imgs.extend(cat_dir.glob(f"*.{ext.upper()}"))

    return sorted(set(imgs))


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return int(round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac))


def compute_repeats(count: int, target: int) -> int:
    if count <= 0:
        return 0
    r = math.ceil(target / count)
    return max(MIN_REPEATS, min(MAX_REPEATS, r))


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT.resolve()}")

    # Categories = direct subfolders under dataset root
    category_dirs = [p for p in DATASET_ROOT.iterdir() if p.is_dir()]
    category_dirs = sorted(category_dirs, key=lambda p: p.name)

    # Count images per category
    cat_images = {}
    cat_counts = {}
    for cat_dir in category_dirs:
        imgs = list_images(cat_dir)
        cat_images[cat_dir.name] = imgs
        cat_counts[cat_dir.name] = len(imgs)

    # Choose balancing target
    vals = sorted([c for c in cat_counts.values() if c > 0])
    
    target = percentile(vals, 0.75)
    if target == 0:
        raise RuntimeError("No images found across categories.")

    # Compute repeats per category
    repeats = {}
    for cat, count in cat_counts.items():
        repeats[cat] = compute_repeats(count, target) if count > 0 else 0

    # Print summary
    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"Target strategy: p75 -> target_per_category={target}")
    print(f"Repeat caps: MIN={MIN_REPEATS}, MAX={MAX_REPEATS}\n")

    total_base = sum(cat_counts.values())
    total_effective = sum(cat_counts[c] * repeats[c] for c in cat_counts)
    print(f"Found {len(category_dirs)} categories")
    print(f"Base images total: {total_base}")
    print(f"Effective rows: {total_effective}\n")

    print("Per-category:")
    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        rep = repeats[cat]
        eff = count * rep
        print(f"  {cat:16s} count={count:4d}  repeats={rep:2d}  effective={eff:5d}")

    # Write metadata.jsonl
    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for cat in sorted(cat_images.keys()):
            imgs = cat_images[cat]
            rep = repeats[cat]
            if rep == 0:
                continue
            for img_path in imgs:
                cap = read_caption(img_path)
                # Keep paths relative to dataset root (recommended)
                rel = img_path.relative_to(DATASET_ROOT).as_posix()
                row = {"file_name": rel, "text": cap}
                for _ in range(rep):
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    print(f"\n✅ Wrote {written} rows to {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

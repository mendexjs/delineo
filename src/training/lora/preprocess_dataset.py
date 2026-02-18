import os
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm 

INPUT_ROOT = Path("raw_dataset")
TARGET_SIZE = 1024
PADDING_COLOR = (255, 255, 255)
OUTPUT_ROOT = Path("dataset/train")


def resize_and_pad(image_path, output_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        # Preserve aspect ratio
        img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Create square background
        new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), PADDING_COLOR)

        # Center image
        x_offset = (TARGET_SIZE - img.width) // 2
        y_offset = (TARGET_SIZE - img.height) // 2

        new_img.paste(img, (x_offset, y_offset))
        new_img.save(output_path, quality=95)


def process_dataset():
    inputs = list(os.walk(INPUT_ROOT))
    for root, dirs, files in tqdm(inputs):
        root_path = Path(root)

        relative_path = root_path.relative_to(INPUT_ROOT)
        target_dir = OUTPUT_ROOT / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = root_path / file
            dst_file = target_dir / file

            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                resize_and_pad(src_file, dst_file)

            elif file.lower().endswith(".txt"):
                shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    process_dataset()
    print("✅ Done! Resized dataset saved to:", OUTPUT_ROOT)

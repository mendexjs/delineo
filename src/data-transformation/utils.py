from PIL import Image
import cv2

TOP_BAR_HEIGHT_TO_CROP = 70
TOP_BAR_HEIGHT_TO_CROP_RESIZED = 48
BOTTOM_NAV_HEIGHT_TO_CROP = 120
BOTTOM_NAV_HEIGHT_TO_CROP_RESIZED = 80

def crop_bars_from_filepath(image_path, resized = False):
    bar_height = TOP_BAR_HEIGHT_TO_CROP_RESIZED if resized else TOP_BAR_HEIGHT_TO_CROP
    nav_height = BOTTOM_NAV_HEIGHT_TO_CROP_RESIZED if resized else BOTTOM_NAV_HEIGHT_TO_CROP
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            if height <= (bar_height + nav_height):
                return img.copy() 
            box = (0, bar_height, width, height - nav_height)
            return img.crop(box)
    except Exception:
        print("Failed to crop image, ")
        return None


def crop_bars_opencv(img, resized = False):
    bar_height = TOP_BAR_HEIGHT_TO_CROP_RESIZED if resized else TOP_BAR_HEIGHT_TO_CROP
    nav_height = BOTTOM_NAV_HEIGHT_TO_CROP_RESIZED if resized else BOTTOM_NAV_HEIGHT_TO_CROP
    if img is None or img.size == 0:
        return None
    height, width, _ = img.shape
    if height > (bar_height + nav_height):
        # Slicing syntax: image[y_start : y_end, x_start : x_end]
        # We crop from Top Bar -> Height minus Bottom Bar
        return img[bar_height : height - nav_height, 0 : width]
    print(f"Skipping crop, image too small.")
    return None

def resize_contain(image, target_w, target_h, interpolation=cv2.INTER_AREA):
    """
    Resizes image to fit *inside* target_w x target_h while maintaining aspect ratio.
    Returns the resized image directly (variable size, NO padding).
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

import json
import os

def load_ui_captions_map(jsonl_path = "./ui_captions_dataset.jsonl"):
    """
    Loads a JSONL file and maps 'filename' -> 'caption' exactly as written.
    """
    captions_map = {}
    
    if not os.path.exists(jsonl_path):
        print(f"Error: Captions file not found at {jsonl_path}")
        return {}

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    entry = json.loads(line)
                    filename = entry.get("filename")
                    caption = entry.get("caption")
                    
                    if filename and caption:
                        captions_map[filename] = caption
                        
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON on line {i+1}")

    except Exception as e:
        print(f"Error reading file: {e}")

    print(f"Loaded {len(captions_map)} captions.")
    return captions_map
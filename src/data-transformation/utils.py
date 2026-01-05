from PIL import Image
import cv2
import json
import os

def image_from_filepath(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            if height < width or width < 720:
                return None
            return img
    except Exception:
        print("Failed to open image")
        return None


def crop_bars_opencv(img, status_height, nav_height):
    if img is None or img.size == 0:
        return None
    height, width, _ = img.shape
    if height > (status_height + nav_height):
        # Slicing syntax: image[y_start : y_end, x_start : x_end]
        # We crop from Top Bar -> Height minus Bottom Bar
        return img[status_height : height - nav_height, 0 : width]
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

def resize_width_and_crop(image, target_w, target_h, interpolation=cv2.INTER_AREA):
    """
    Resizes image to a fixed target_w while maintaining aspect ratio.
    If the resulting height exceeds target_h, the bottom is cropped.
    """
    h, w = image.shape[:2]
    scale = target_w / w
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (target_w, new_h), interpolation=interpolation)
    
    # Cut height if too tall (Top-Crop: keeps the top, cuts the bottom)
    if new_h > target_h:
        resized = resized[0:target_h, :]
        
    return resized


STATUS_BAR_HEIGHT_ANDROID = 42
STATUS_BAR_HEIGHT_IPHONE = 34
STATUS_BAR_HEIGHT_TALL_IPHONE = 50
STATUS_BAR_HEIGHT_RICO = 64
def crop_vins_status_bar(img, platform, orig_proportion):
    if img is None:
        return None
    
    height, width, _ = img.shape
    
    crop_top = 0
    if platform == "Android":
        crop_top = STATUS_BAR_HEIGHT_ANDROID
    elif platform == "iphone":
        if orig_proportion < 9/16:
            # 9:19.5 screens will be scaled to fit 720px width and statusbar gets up to 50px at the top
            crop_top = STATUS_BAR_HEIGHT_TALL_IPHONE
        else:
            crop_top = STATUS_BAR_HEIGHT_IPHONE
    elif platform == "Rico":
        crop_top = STATUS_BAR_HEIGHT_RICO
        
    # Crop [y:h, x:w]
    return img[crop_top:height, 0:width]

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
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import random
from utils import crop_vins_status_bar, resize_width_and_crop

# --- CONFIGURATION ---
NUM_CPUS = -1 # All CPUs available
current_directory = os.path.dirname(os.path.abspath(__file__))
VINS_ROOT = Path(os.path.join(current_directory, "../raw-data/vins")) 

OUTPUT_TRAIN_DIR = Path("/scratch/delineo_data/train/vins")
OUTPUT_VALIDATION_DIR = Path("/scratch/delineo_data/validation")
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VALIDATION_DIR, exist_ok=True)

# Add IDs here if you want specific validation split
VALIDATION_SAMPLES = (
    'Android_2',
    'IMG_1517',
    'IMG_0751',
)

BG_COLOR = (0, 0, 0)
CONTRAST_COLOR = (255, 255, 255)
AVG_LINE_HEIGHT_PIXELS = 40
MIN_SEMANTIC_ELEMENTS = 3
MAX_SEMANTIC_ELEMENTS = 30 
STROKE_WIDTH = 3

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280


def calculate_lines(height):
    if height <= 0:
        return 1
    total_lines = int(round(height / AVG_LINE_HEIGHT_PIXELS))
    return max(1, total_lines)

def draw_filled_rectangle(img, bounds):
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=BG_COLOR, thickness=-1)
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_image_placeholder(img, bounds):
    draw_filled_rectangle(img, bounds)
    cv2.line(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)
    cv2.line(img, (bounds[0], bounds[3]), (bounds[2], bounds[1]), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_icon_placeholder(img, bounds):
    x_center = int((bounds[0] + bounds[2]) / 2)
    y_center = int((bounds[1] + bounds[3]) / 2)
    center_point = (x_center, y_center)
    box_width = bounds[2] - bounds[0]
    box_height = bounds[3] - bounds[1]
    radius = int((min(box_width, box_height) / 2) * 1.1)
    
    if radius < 5: return

    cv2.circle(img, center_point, radius, color=BG_COLOR, thickness=-1)
    cv2.circle(img, center_point, radius, color=CONTRAST_COLOR, thickness=STROKE_WIDTH)
    
    padding = int(radius * 0.4) 
    x1_in = x_center - radius + padding
    y1_in = y_center - radius + padding
    x2_in = x_center + radius - padding
    y2_in = y_center + radius - padding
    
    cv2.line(img, (x1_in, y1_in), (x2_in, y2_in), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)
    cv2.line(img, (x1_in, y2_in), (x2_in, y1_in), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_text_placeholder(img, bounds):
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    
    if height < 5 or width < 5: return

    num_lines = calculate_lines(height)
    padding_y = max(int(height * 0.1), 5)
    available_height = height - (2 * padding_y)
    line_spacing = available_height / max(1, num_lines)
    
    line_start = bounds[0]
    line_end = bounds[2]
    
    for i in range(num_lines):
        y_pos = bounds[1] + padding_y + int(i * line_spacing) + int(line_spacing/2)
        if y_pos >= bounds[3]: break
        cv2.line(img, (line_start, y_pos), (line_end, y_pos), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_container_placeholder(img, bounds):
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_button_placeholder(img, bounds):
    draw_filled_rectangle(img, bounds)
    x1, y1, x2, y2 = bounds
    width = x2 - x1
    height = y2 - y1

    padding_x = int(width * 0.15)
    line_start_x = x1 + padding_x
    line_end_x = x2 - padding_x
    line_y = y1 + (height // 2)

    if line_end_x > line_start_x:
        cv2.line(img, (line_start_x, line_y), (line_end_x, line_y), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

def draw_checkbox_placeholder(img, bounds):
    draw_filled_rectangle(img, bounds)
    cv2.line(img, (bounds[0]+2, bounds[1]+(bounds[3]-bounds[1])//2), (bounds[0]+(bounds[2]-bounds[0])//2, bounds[3]-2), color=CONTRAST_COLOR, thickness=STROKE_WIDTH)

# --- MAPPING LOGIC ---

VISUAL_FUNCS = {
    "Image": draw_image_placeholder,
    "Video": draw_image_placeholder,
    "Icon": draw_icon_placeholder,
    "Text": draw_text_placeholder,
    "Button": draw_button_placeholder,
    "Container": draw_container_placeholder,
    "Checkbox": draw_checkbox_placeholder,
    "Box": draw_filled_rectangle,
}

CLASS_TO_VISUAL = {
    # Basic Text
    'Text': 'Text',
    'CheckedTextView': 'Text',
    
    # Buttons
    'TextButton': 'Button',
    'Button': 'Button',
    
    # Boxes
    'EditText': 'Box',
    'PageIndicator': 'Box',
    
    # Icons/Images
    'Icon': 'Icon',
    'Spinner': 'Icon',
    'Image': 'Image',
    
    # Controls
    'CheckedTextView': 'Checkbox',
    'CheckBox': 'Checkbox',
    'Checkbox': 'Checkbox',
    'Switch': 'Checkbox',
    
    'Drawer': 'Container',
    'Modal': 'Container',
}

def get_noisy_transformer(alpha, sigma):
  return A.Compose([
    A.ElasticTransform(alpha=alpha, sigma=sigma, p=1, border_mode=cv2.BORDER_CONSTANT),
    A.CoarseDropout(
     num_holes_range=(3, 5),
     hole_height_range=(5, 15),
     hole_width_range=(5, 20),
     fill="inpaint_ns",
     p=0.3
 )
])

# --- VINS DATA PARSING ---

def parse_vins_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            objects.append({
                'class': name,
                'bounds': [xmin, ymin, xmax, ymax],
                'text': None 
            })
            
        return width, height, objects
    except Exception as e:
        return None, None, None

def get_all_vins_files():
    data_pairs = []
    platforms = ["Android", "iphone", "Rico"]
    
    for platform in platforms:
        ann_dir = VINS_ROOT / platform / "Annotations"
        img_dir = VINS_ROOT / platform / "JPEGImages"
        
        if not ann_dir.exists() or not img_dir.exists():
            continue
            
        xml_files = list(ann_dir.glob("*.xml"))
        for xml_file in xml_files:
            file_id = xml_file.stem
            img_path = img_dir / f"{file_id}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{file_id}.png"
            
            if img_path.exists():
                data_pairs.append({
                    'id': file_id,
                    'platform': platform,
                    'xml_path': str(xml_file),
                    'img_path': str(img_path)
                })
    return data_pairs

def validate_single_file(file_info):
    width, height, objects = parse_vins_xml(file_info['xml_path'])
    
    if objects is None: return None
        
    mapped_count = 0
    valid_objects = []
    
    for obj in objects:
        cls_name = obj['class']
        if cls_name in CLASS_TO_VISUAL:
            mapped_count += 1
            valid_objects.append(obj)
            
    if mapped_count < MIN_SEMANTIC_ELEMENTS or mapped_count > MAX_SEMANTIC_ELEMENTS:
        return None
        
    return {
        'id': file_info['id'],
        'platform': file_info['platform'],
        'img_path': file_info['img_path'],
        'width': width,
        'height': height,
        'views': valid_objects
    }

def get_valid_input_data():
    raw_files = get_all_vins_files()
    print(f"--- FILTERING VALID INPUT DATA (PARALLEL - {len(raw_files)} files) ---")
    
    results = Parallel(n_jobs=NUM_CPUS, backend="loky")(
        delayed(validate_single_file)(f) for f in tqdm(raw_files, desc="Validating XMLs")
    )

    valid_files = [res for res in results if res is not None]
    print(f"✅ {len(valid_files)} valid files loaded.")
    return valid_files

# --- PROCESSING ---

def process_single_item(item):
    sample_id = item['id']
    platform = item['platform']

    try:
        # 1. Load Image
        ui_img = cv2.imread(item['img_path'])
        if ui_img is None: 
            return False
        
        ui_height, ui_width, _ = ui_img.shape
        if ui_width > ui_height or ui_width < TARGET_WIDTH: 
            return False 

        # 2. Draw Wireframe
        width = item['width']
        height = item['height']
        
        output_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        for obj in item['views']:
            visual_key = obj['class']
            bounds = obj['bounds']
            visual_type = CLASS_TO_VISUAL.get(visual_key, 'Container')
            
            if visual_type in VISUAL_FUNCS:
                VISUAL_FUNCS[visual_type](output_canvas, bounds)

        orig_proportion = width/height
        # UI: Smooth interpolation
        ui_final = crop_vins_status_bar(resize_width_and_crop(
            ui_img, 
            TARGET_WIDTH, TARGET_HEIGHT, 
            interpolation=cv2.INTER_AREA
        ), platform, orig_proportion)
        
        # Wireframe: Sharp interpolation
        wireframe_final = crop_vins_status_bar(resize_width_and_crop(
            output_canvas, 
            TARGET_WIDTH, TARGET_HEIGHT, 
            interpolation=cv2.INTER_NEAREST
        ), platform, orig_proportion)

        if ui_final is None or wireframe_final is None:
            return False

        # 4. Augmentations
        (alpha, sigma) = (random.randint(300, 400), random.randint(20, 25))
        transform_humanize = get_noisy_transformer(alpha, sigma)
        
        augmented = transform_humanize(image=wireframe_final)
        canvas_with_noise = augmented['image']

        # 5. Export
        export_base_path = OUTPUT_VALIDATION_DIR if sample_id in VALIDATION_SAMPLES else OUTPUT_TRAIN_DIR
        
        # Save X (Input Sketch)
        output_path_x = os.path.join(export_base_path, f"{platform}_{sample_id}_input.png")
        canvas_bgr = cv2.cvtColor(canvas_with_noise, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path_x, canvas_bgr)

        # Save Y (Target UI)
        output_path_y = os.path.join(export_base_path, f"{platform}_{sample_id}_output.png")
        cv2.imwrite(output_path_y, ui_final)
        
        return True

    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return False


def main():
    if not VINS_ROOT.exists():
        print(f"❌ VINS_ROOT not found at: {VINS_ROOT}")
        return

    filtered_data = get_valid_input_data()

    SAMPLE_BATCH_SIZE = len(filtered_data)
    input_batch = random.sample(filtered_data, SAMPLE_BATCH_SIZE)

    print(f"--- PROCESSING {len(input_batch)} DATA ITEMS IN PARALLEL ---")
    results = Parallel(n_jobs=NUM_CPUS, verbose=0)(
        delayed(process_single_item)(item) for item in tqdm(input_batch, desc="Processing Items")
    )

    processed_count = sum(results)
    skipped_count = len(input_batch) - processed_count

    print("\n--- DATA BATCH PROCESSING CONCLUDED ---")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
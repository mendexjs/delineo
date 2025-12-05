import os
import json
import cv2
import numpy as np
from PIL import Image
import albumentations as A
import random
import math
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
FILES_ROOT = "/Users/matheus/Downloads"
DATA_ROOT = os.path.join(FILES_ROOT, "combined")
ANNOTATIONS_ROOT = os.path.join(FILES_ROOT, "semantic_annotations")
SKETCHES_ROOT = os.path.join(FILES_ROOT, "sketches")

OUTPUT_DIR = os.path.join(current_directory, "outputs")

OUTPUT_DIR_X = os.path.join(OUTPUT_DIR, "train_X_sketches")
os.makedirs(OUTPUT_DIR_X, exist_ok=True)

# Components that we want to avoid in our training dataset
FORBIDDEN_COMPONENTS = {
    "Web View", "WebView",
    "Date Picker", "DatePicker",
    "Calendar", "CalendarView",
    "Time Picker", "TimePicker",
    "Map", "MapView", "Map View",
    "Ad", "Advertisement"
    "Drawer",
    "Modal"
}

def check_forbidden_components(node):
    label = node.get('componentLabel')

    if label in FORBIDDEN_COMPONENTS:
        return True

    if 'children' in node and node['children']:
        for child in node['children']:
            if check_forbidden_components(child):
                return True

    return False


BG_COLOR = (0, 0, 0)
CONTRAST_COLOR = (255, 255, 255)

AVG_CHAR_WIDTH_PIXELS = 50
def calculate_lines(text_content, width):
    if not text_content or width <= 0:
        return 1

    max_chars_per_line = max(1, int((width - 20) / AVG_CHAR_WIDTH_PIXELS))
    explicit_lines = text_content.split('\n')
    total_lines = 0
    for line in explicit_lines:
        wrapped_lines = max(1, math.ceil(len(line) / max_chars_per_line))
        total_lines += wrapped_lines

    return max(1, total_lines)

def draw_image_placeholder(img, bounds):
    """Draw an X inside a rectangle to represent an image"""
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=BG_COLOR, thickness=-1)
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)
    cv2.line(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)
    cv2.line(img, (bounds[0], bounds[3]), (bounds[2], bounds[1]), color=CONTRAST_COLOR, thickness=3)

def draw_icon_placeholder(img, bounds):
    """Draw a circle to represent an icon"""
    x_center = int((bounds[0] + bounds[2]) / 2)
    y_center = int((bounds[1] + bounds[3]) / 2)
    center_point = (x_center, y_center)

    box_width = bounds[2] - bounds[0]
    box_height = bounds[3] - bounds[1]
    radius = int(min(box_width, box_height) / 2)

    cv2.circle(img, center_point, radius, color=CONTRAST_COLOR, thickness=3)

def draw_text_placeholder(img, bounds, text_content=None):
    """Draw horizontal lines to represent text"""
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]

    if height < 5 or width < 5:
        return

    num_lines = calculate_lines(text_content, width)
    padding_y = max(int(height * 0.1), math.ceil(AVG_CHAR_WIDTH_PIXELS/2))
    line_spacing = (height - 2 * padding_y) / max(1, (num_lines - 1)) if num_lines > 1 else 0

    for i in range(num_lines):
        y_pos = bounds[1] + padding_y + int(i * line_spacing)
        # Centralize if only one line
        if num_lines == 1:
            y_pos = bounds[1] + int(height / 2)
        # Avoid drawing beyond bounds
        if y_pos >= bounds[3] - padding_y and i > 0:
            break

        cv2.line(img, (bounds[0], y_pos), (bounds[2], y_pos), color=CONTRAST_COLOR, thickness=2)

def draw_container_placeholder(img, bounds):
    """Draw a rectangle to represent a container"""
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=BG_COLOR, thickness=-1)
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)

def draw_button_placeholder(img, bounds, text_content=None):
    """Drawn a rectangle with text lines inside"""
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)
    inner_bounds = [bounds[0] + 80, bounds[1] + 20, bounds[2] - 80, bounds[3] - 20]
    if inner_bounds[2] > inner_bounds[0] and inner_bounds[3] > inner_bounds[1]:
        draw_text_placeholder(img, inner_bounds, text_content)

VISUAL_LANGUAGE = {
    "Image": draw_image_placeholder,
    "Video": draw_image_placeholder,
    "Icon": draw_icon_placeholder,
    "Text": draw_text_placeholder,
    "Text Button": draw_button_placeholder,
    "Input": draw_button_placeholder,
    "Button": draw_button_placeholder,
    "Toolbar": draw_container_placeholder,
    "Multi-Tab": draw_container_placeholder,
    "Card": draw_container_placeholder,
    "Checkbox": draw_container_placeholder,
}


def traverse_and_draw(node, canvas_array):
    """Recursively traverse the semantic map and draw components on the canvas"""
    label = node.get('componentLabel')
    text_content = node.get('text')
    bounds = node.get('bounds')

    if label in VISUAL_LANGUAGE and bounds and (bounds[2] - bounds[0] > 5) and (bounds[3] - bounds[1] > 5):
        if label in ["Text", "Input", "Button"]:
            VISUAL_LANGUAGE[label](canvas_array, bounds, text_content)
        else:
            VISUAL_LANGUAGE[label](canvas_array, bounds)

    if 'children' in node and node['children']:
        for child in node['children']:
            traverse_and_draw(child, canvas_array)


def get_noisy_transformer(alpha, sigma):
  """Create an Albumentations transformer that adds noise and distortions to the image."""
  return A.Compose([
    # Add wavy distortions
    A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    # Rotate slightly
    A.Rotate(limit=0.5, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    # Add random coarse dropout (like smudges or missing parts)
    A.CoarseDropout(
     num_holes_range=(5, 20),
     hole_height_range=(10, 30),
     hole_width_range=(20, 50),
     fill="inpaint_ns",
     p=0.4
 )
])

MIN_SEMANTIC_ELEMENTS = 4
all_json_files = [f for f in os.listdir(ANNOTATIONS_ROOT) if f.endswith('.json')]

def count_elements(node):
  elements_count = 0
  if 'componentLabel' in node:
      elements_count += 1

  if 'children' in node and node['children']:
      for child in node['children']:
          elements_count += count_elements(child)
  return elements_count

def get_valid_input_datum(sample_id, verbose=False):
  json_path = os.path.join(ANNOTATIONS_ROOT, f"{sample_id}.json")
  annotations_img_path = os.path.join(ANNOTATIONS_ROOT, f"{sample_id}.png")
  real_ui_path = os.path.join(DATA_ROOT, f"{sample_id}.jpg")
  try:
    if not os.path.exists(annotations_img_path):
        if verbose:
          print(f"File {annotations_img_path} does not exist")
        return (False, None)
    if not os.path.exists(real_ui_path):
        if verbose:
          print(f"File {real_ui_path} does not exist")
        return (False, None)

    with open(json_path, 'r', encoding='utf-8') as f:
        semantic_map = json.load(f)

    elements_count = count_elements(semantic_map)
    if elements_count < MIN_SEMANTIC_ELEMENTS:
      if verbose:
          print(f"Sample ID {sample_id} skipped because has less than {MIN_SEMANTIC_ELEMENTS} elements")
      return (False, None)

    if check_forbidden_components(semantic_map):
        if verbose:
          print(f"Sample ID {sample_id} has forbidden components")
        return (False, None)
  except Exception as e:
    if verbose:
        print(f"Error processing Sample ID {sample_id}: {e}")
    return (False, None)

  return (True, semantic_map)


def get_valid_input_data(sample_size=None, verbose=False):
  """Filter and return valid input data samples."""
  files = all_json_files
  if sample_size:
    files = all_json_files[:sample_size]
  print("--- FILTERING VALID INPUT DATA ---")
  valid_files = []
  for file in tqdm(files):
    sample_id = file.replace(".json", "")
    (valid, semantic_map) = get_valid_input_datum(sample_id, verbose)
    if valid:
      valid_files.append({ "id": sample_id, "semantic_map": semantic_map})

  print(f"{len(valid_files)} valid files. {len(files) - len(valid_files)} files skipped")

  return valid_files

print(f"Total of {len(all_json_files)} RICO UI examples found.")
filtered_overall_data = get_valid_input_data()
print(f"Total of {len(filtered_overall_data)} examples valid for training.")

sample_size = 100
processed_count = 0
skipped_count = 0
input_data = random.sample(filtered_overall_data, sample_size)

print("--- PROCESSING DATA BATCH ---")
for input in tqdm(input_data):
    try:
        sample_id = input['id']
        semantic_map = input['semantic_map']
        annotations_img_path = os.path.join(ANNOTATIONS_ROOT, f"{sample_id}.png")
        annotations_img = cv2.imread(annotations_img_path)
        height, width, _ = annotations_img.shape
        # filling with zeros because background should be black
        output_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        traverse_and_draw(semantic_map, output_canvas)

        (alpha, sigma) = (random.randint(500, 600), random.randint(20, 30))
        transform_humanize = get_noisy_transformer(alpha, sigma)
        augmented = transform_humanize(image=output_canvas)
        canvas_with_noise = augmented['image']

        output_path_x = os.path.join(OUTPUT_DIR_X, f"{sample_id}.png")
        Image.fromarray(canvas_with_noise).save(output_path_x)

        processed_count += 1

    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        skipped_count += 1
        continue

print("\n--- DATA BATCH PROCESSING CONCLUDED ---")
print(f"✅ Successfully processed: {processed_count}. Exported to {OUTPUT_DIR_X}")
print(f"❌ Skipped examples (Missing, filtered out or corrupted): {skipped_count}")
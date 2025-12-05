import os
import json
import re
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
import math
import random

# --- CONFIGURATION ---
current_directory = os.path.dirname(os.path.abspath(__file__))
FILES_ROOT = "/Users/matheus/Downloads" # Update this to your MUD dataset location
MUD_ROOT = os.path.join(FILES_ROOT, "mud-dataset") # Folder containing .jpg / .png

OUTPUT_DIR = os.path.join(current_directory, "mud_outputs")
OUTPUT_DIR_X = os.path.join(OUTPUT_DIR, "train_X_sketches")
os.makedirs(OUTPUT_DIR_X, exist_ok=True)

# --- DRAWING UTILS ---
BG_COLOR = (0, 0, 0)
CONTRAST_COLOR = (255, 255, 255)
AVG_CHAR_WIDTH_PIXELS = 13

def get_class_suffix(node):
    full_class = node.get('class', '')
    return full_class.split('.')[-1]

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

def draw_filled_rectangle(img, bounds, stroke=3):
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=BG_COLOR, thickness=-1)
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=stroke)

def draw_image_placeholder(img, bounds, text_content=None):
    """Draw an X inside a rectangle"""
    draw_filled_rectangle(img, bounds)
    cv2.line(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)
    cv2.line(img, (bounds[0], bounds[3]), (bounds[2], bounds[1]), color=CONTRAST_COLOR, thickness=3)

def draw_icon_placeholder(img, bounds, text_content=None):
    """Draw a circle with an 'X' inside to represent an icon."""
    # Calculate Center and Radius for the Circle
    x_center = int((bounds[0] + bounds[2]) / 2)
    y_center = int((bounds[1] + bounds[3]) / 2)
    center_point = (x_center, y_center)
    box_width = bounds[2] - bounds[0]
    box_height = bounds[3] - bounds[1]
    radius = int((min(box_width, box_height) / 2) * 0.8) # 20% padding
    
    if radius < 5:
        return

    cv2.circle(img, center_point, radius, color=BG_COLOR, thickness=-1)
    cv2.circle(img, center_point, radius, color=CONTRAST_COLOR, thickness=3)

    # Determine the inner area for the 'X' (e.g., 60% of the radius)
    padding = int(radius * 0.4) 
    
    # Define the coordinates for the two intersecting lines (X)
    x1_in = x_center - radius + padding
    y1_in = y_center - radius + padding
    x2_in = x_center + radius - padding
    y2_in = y_center + radius - padding
    
    cv2.line(img, (x1_in, y1_in), (x2_in, y2_in), color=CONTRAST_COLOR, thickness=3)
    cv2.line(img, (x1_in, y2_in), (x2_in, y1_in), color=CONTRAST_COLOR, thickness=3)

def draw_text_placeholder(img, bounds, text_content=None, fit_text=None, center_text=None):
    """Draw horizontal lines representing text. Single lines reflect content width; multi-lines span the container width."""
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    
    if height < 5 or width < 5: 
        return

    num_lines = calculate_lines(text_content, width)
    padding_y = max(int(height * 0.1), 5)
    
    available_height = height - (2 * padding_y)
    line_spacing = available_height / max(1, num_lines)
    
    if num_lines == 1 and fit_text and text_content:
        # Calculate simulated pixel width based on character count
        simulated_text_width = len(text_content) * AVG_CHAR_WIDTH_PIXELS
        
        effective_line_width = min(simulated_text_width, width)
        line_start = bounds[0]
        if center_text:
            line_start = bounds[0] + int((width - effective_line_width) / 2)

        line_end = line_start + effective_line_width
        
    else:
        # --- Multi-Line Logic: Span nearly the full container width ---
        # All lines will have the same length
        line_start = bounds[0]
        line_end = bounds[2]
    
    for i in range(num_lines):
        # Calculate the Y position for the center of the line
        y_pos = bounds[1] + padding_y + int(i * line_spacing) + int(line_spacing/2)
        
        if y_pos >= bounds[3]: 
            break
        
        cv2.line(img, (line_start, y_pos), (line_end, y_pos), color=CONTRAST_COLOR, thickness=2)

def draw_container_placeholder(img, bounds, text_content=None):
    """Draw a rectangle outline"""
    cv2.rectangle(img, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color=CONTRAST_COLOR, thickness=3)

def draw_button_placeholder(img, bounds, text_content=None):
    """Rectangle with centralized text lines inside"""
    padding = 20
    padding_bounds = [bounds[0] + padding, bounds[1], bounds[2] - padding, bounds[3]]
    draw_filled_rectangle(img, padding_bounds)
    # Inner padding for text
    inner_bounds = [padding_bounds[0] + padding, padding_bounds[1] + padding, padding_bounds[2] - padding, padding_bounds[3] - padding]
    if text_content and inner_bounds[2] > inner_bounds[0] and inner_bounds[3] > inner_bounds[1]:
        draw_text_placeholder(img, inner_bounds, text_content, fit_text=True, center_text=True)

def draw_input_placeholder(img, bounds, text_content=None):
    """Rectangle with left-justified text lines inside"""
    draw_filled_rectangle(img, bounds)
    # Inner padding for text
    inner_bounds = [bounds[0] + 50, bounds[1], bounds[2], bounds[3]]
    if text_content and inner_bounds[2] > inner_bounds[0] and inner_bounds[3] > inner_bounds[1]:
        draw_text_placeholder(img, inner_bounds, text_content, fit_text=True)

def draw_checkbox_placeholder(img, bounds, text_content=None):
    """Small square with potential check"""
    draw_filled_rectangle(img, bounds, stroke=2)
    # Draw a small 'tick' simulation
    cv2.line(img, (bounds[0]+2, bounds[1]+(bounds[3]-bounds[1])//2), (bounds[0]+(bounds[2]-bounds[0])//2, bounds[3]-2), color=CONTRAST_COLOR, thickness=2)

# --- MAPPING LOGIC ---

VISUAL_FUNCS = {
    "Image": draw_image_placeholder,
    "Video": draw_image_placeholder,
    "Icon": draw_icon_placeholder,
    "Text": draw_text_placeholder,
    "Button": draw_button_placeholder,
    "Input": draw_input_placeholder,
    "Container": draw_container_placeholder,
    "Checkbox": draw_checkbox_placeholder,
}

CLASS_TO_VISUAL = {
    # Text
    'TextView': 'Text',
    
    # Inputs/Interactables
    'EditText': 'Input',
    'KCheckBox': 'Checkbox',
    'Switch': 'Checkbox',
    'ToggleButton': 'Checkbox',
    'CompoundButton': 'Checkbox',
    
    # Buttons
    'Button': 'Button',
    'MenuItem': 'Button',
    
    # Media
    'ImageView': 'Image',
    'Image': 'Image',
    'VideoView': 'Video',
    'MutedVideoView': 'Video',
    'ImageButton': 'Icon',
    
    # Containers / Layouts
    'CardView': 'Container',
    'MaterialCardView': 'Container',
    'View': 'Container',
    'SidebarLayout': 'Container',
    'Gallery': 'Container',
    'ViewFlipper': 'Container',
    'TabWidget': 'Container',
    'DrawerLayout': 'Container',
    'Dialog': 'Container',
}

FORBIDDEN_CLASSES = {
    "WebView", "Webview", "WebViewComponent",
    "Calendar", "CalendarView",
    "DatePicker", "TimePicker",
    "MapView", "Map",
    "AdView", "AdBanner", "Advertisement",
    "Dialog", "AlertDialog"
    "TableLayout"
}

# Pattern to find any character NOT in the allowed set (Latin, numbers, symbols, etc.)
# Avoiding UIs with other kinds of alphabets to avoid noise in text representation  ‘Like A Kid’
# ----------------------------------------------------------------------------------
latin_re = re.compile(
    r"""^[\sA-Za-z0-9/;:\.,\(\)\{\}\[\]_+=!@?#\$£˚…€%&\*\|'"<>\-
        \u00C0-\u00FF\u0100-\u017F\u2010-\u205E
    ]*$""",
    re.UNICODE | re.VERBOSE
)

def contains_forbidden_non_latin(text_content):
    """
    Checks if a string contains any character outside the basic Latin, 
    numbers, and common punctuation set.
    """
    if not isinstance(text_content, str) or not text_content:
        return False
        
    return latin_re.fullmatch(text_content) is None

def check_forbidden_components_and_text(all_views, verbose=False):
    """
    Checks the flat list of views for forbidden class suffixes AND invalid text content.
    Returns True if the sample should be filtered out.
    """
    if not all_views:
        return False
        
    for node in all_views:
        class_suffix = get_class_suffix(node)
        
        if class_suffix in FORBIDDEN_CLASSES:
            if verbose:
                print(f"  [Filter]: Forbidden class found: {class_suffix}")
            return True 
        
        text_content = node.get('text')        
        if text_content and isinstance(text_content, str) and contains_forbidden_non_latin(text_content):
            if verbose:
                print(f"[Filter]: Invalid non-Latin text found in node. {text_content}")
            return True

    return False

def traverse_and_draw(view_idx, all_views, canvas_array):
    """
    Recursively traverse MUD structure.
    view_idx: Integer index of current view in all_views list
    all_views: List of view dictionaries
    """
    try:
        node = all_views[view_idx]
    except IndexError:
        return

    # Check Visibility
    if not node.get('visible', True):
        return

    class_suffix = get_class_suffix(node)
    
    # Handle Bounds (MUD format: [[x1, y1], [x2, y2]])
    bounds_raw = node.get('bounds')
    if bounds_raw:
        x1, y1 = bounds_raw[0]
        x2, y2 = bounds_raw[1]
        width = x2 - x1
        height = y2 - y1
        flat_bounds = [int(x1), int(y1), int(x2), int(y2)]

        # Draw logic
        if width > 5 and height > 5:
            # Check if this class is in our mapping list
            if class_suffix in CLASS_TO_VISUAL or node.get('clickable'):
                visual_type = CLASS_TO_VISUAL.get(class_suffix, 'Button')
                text_content = node.get('text')
                
                # Execute drawing
                if visual_type in VISUAL_FUNCS:
                    VISUAL_FUNCS[visual_type](canvas_array, flat_bounds, text_content)

    # Recursion: 'children' in MUD is a list of integers (indices)
    children_indices = node.get('children', [])
    for child_idx in children_indices:
        traverse_and_draw(child_idx, all_views, canvas_array)


def get_noisy_transformer(alpha, sigma):
  return A.Compose([
    A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    A.Rotate(limit=0.5, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.CoarseDropout(
     num_holes_range=(5, 20),
     hole_height_range=(10, 30),
     hole_width_range=(20, 50),
     fill="inpaint_ns",
     p=0.4
 )
])

# --- DATA PROCESSING ---

all_json_files = [f for f in os.listdir(MUD_ROOT) if f.endswith('.json')]
MIN_SEMANTIC_ELEMENTS = 3

def count_flat_mapped_elements(all_views):
    mapped_count = 0
    for node in all_views:
        if not node.get('visible', False):
            continue

        class_suffix = get_class_suffix(node)        
        if class_suffix in CLASS_TO_VISUAL:
            mapped_count += 1

    return mapped_count

def get_valid_input_datum(sample_id, verbose=False):
  json_path = os.path.join(MUD_ROOT, f"{sample_id}.json")
  img_path_png = os.path.join(MUD_ROOT, f"{sample_id}.png")

  try:
    if not os.path.exists(json_path):
      return (False, None)

    if not os.path.exists(img_path_png):
      if verbose: 
        print(f"Image not found for {sample_id}")
      return (False, None)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    views_list = data['views']

    elements_count = count_flat_mapped_elements(views_list)
    if elements_count < MIN_SEMANTIC_ELEMENTS:
      if verbose:
        print(f"Sample ID {sample_id} skipped because has less than {MIN_SEMANTIC_ELEMENTS} mapped elements")
      return (False, None)
    
    if check_forbidden_components_and_text(views_list, verbose):
        return (False, None)

    return (True, data)

  except Exception as e:
    if verbose:
      print(f"Error processing Sample ID {sample_id}: {e}")
    return (False, None)

def get_valid_input_data(sample_size=None):
  files = all_json_files
  if sample_size:
    files = all_json_files[:sample_size]
  
  print("--- FILTERING VALID INPUT DATA ---")
  valid_files = []
  for file in tqdm(files):
    sample_id = file.replace(".json", "")
    (valid, semantic_data) = get_valid_input_datum(sample_id, verbose=False)
    if valid:
      valid_files.append({ "id": sample_id, "data": semantic_data})
  
  print(f"{len(valid_files)} valid files.")
  return valid_files


# --- MAIN EXECUTION ---

def main():

    filtered_data = get_valid_input_data()

    # Process a batch
    SAMPLE_BATCH_SIZE = min(100, len(filtered_data))
    input_batch = random.sample(filtered_data, SAMPLE_BATCH_SIZE)

    print("--- PROCESSING DATA BATCH ---")
    processed_count = 0
    skipped_count = 0

    for item in tqdm(input_batch):
        try:
            sample_id = item['id']
            mud_data = item['data']
            
            # Dimensions from JSON
            width = int(mud_data.get('width', 1080))
            height = int(mud_data.get('height', 1920))
            
            # Create blank canvas (Black background)
            output_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # MUD 'views' is a flat list. Usually index 0 is the root.
            # We pass the list and the starting index.
            views_list = mud_data['views']
            if views_list:
                traverse_and_draw(0, views_list, output_canvas)

            # Albumentations (Humanize)
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
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Skipped: {skipped_count}")


if __name__ == "__main__":
    print(f"Total of {len(all_json_files)} MUD UI examples found.")
    main()
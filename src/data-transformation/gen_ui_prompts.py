import os
import json
import time
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from joblib import Parallel, delayed

# --- Configuration ---
API_KEY = "YOUR_GOOGLE_API_KEY"
IMAGE_DIRECTORY = "./my_ui_images"
OUTPUT_JSON_FILE = "ui_captions_dataset.json"

TOP_BAR_HEIGHT_TO_CROP = 70
BOTTOM_NAV_HEIGHT_TO_CROP = 120
N_JOBS = 8 
BATCH_SIZE = 50  # Save progress after every 50 images

# --- System Prompt & Model Setup ---
genai.configure(api_key=API_KEY)

# Limiting 70 tokens to keep descriptions concise and avoid get truncated by model's encoder
SYSTEM_INSTRUCTION = """
You are an AI assistant creating captions for training a Stable Diffusion model focused on mobile UI design.
Your task is to look at the provided mobile UI screenshot and generate a concise, high-fidelity description.

Guidelines:
1.  **Strict Output:** Return ONLY the description. Do not use Markdown code blocks. Do not say "Here is a description".
2.  **Start phrase:** Always start with "High-fidelity mobile UI design of...".
3.  **Identify the screen:** Define the type of page (e.g., "a login screen", "a search results feed").
4.  **Key Components:** Describe layout structure and hierarchy (e.g., "vertical list of cards", "tab bar at bottom").
5.  **Visual Details:** Mention specific styles like "rounded corners", "bold typography", "outlined icons".
6.  **Aesthetic:** Use style tokens like "clean", "modern", "minimalist", or "vibrant".
7.  **Length:** Keep the total description under 70 tokens. Do not over-describe; focus on key visual elements.
8.  **Brand Names:** Avoid specific brand names. Instead of "Google logo", say "a multi-colored 'G' logo".
"""

# We initialize the model configuration once, but we will instantiate the model
# inside the thread/process if needed, though for threads it's often safe globally.
# For joblib 'threading' backend, a global model instance usually works.
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=70,
        temperature=0.3,
    ),
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# --- Helper Functions ---

def crop_bars(image_path):
    """Opens and crops image. Returns PIL Image or None."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            if height <= (TOP_BAR_HEIGHT_TO_CROP + BOTTOM_NAV_HEIGHT_TO_CROP):
                return img.copy() 
            box = (0, TOP_BAR_HEIGHT_TO_CROP, width, height - BOTTOM_NAV_HEIGHT_TO_CROP)
            return img.crop(box)
    except Exception as e:
        # print(f"Error cropping {image_path}: {e}") # Optional: silent fail to keep progress bar clean
        return None

def process_single_image(filename):
    """
    Worker function to process a single image. 
    Returns a dict {filename, caption} or None if failed.
    """
    file_path = os.path.join(IMAGE_DIRECTORY, filename)
    
    cropped_image = crop_bars(file_path)
    if not cropped_image:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(cropped_image)
            # Strict cleaning: remove potential markdown or extra spaces
            caption = response.text.strip().replace("```", "")
            return {
                "filename": filename,
                "caption": caption
            }
        except Exception as e:
            if "429" in str(e): # Rate limit error
                time.sleep(2 * (attempt + 1)) # Exponential backoff
            else:
                # If it's not a rate limit (e.g. 500 error), break and return None
                break
    
    return None # Failed after retries

def save_results(data, filename):
    """Saves list of data to JSON file."""
    # Write to a temporary file first, then rename to ensure atomicity 
    # (prevents corrupt files if script crashes while writing)
    temp_filename = filename + ".tmp"
    with open(temp_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_filename, filename)


def main():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    all_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(all_files)} images. Processing with {N_JOBS} parallel threads...")

    results_data = []
    
    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i : i + BATCH_SIZE]
        
        batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(process_single_image)(f) for f in tqdm(batch_files, desc=f"Batch {i//BATCH_SIZE + 1}", leave=False)
        )
        
        valid_results = [res for res in batch_results if res is not None]
        results_data.extend(valid_results)
        
        # Save progress after batch
        save_results(results_data, OUTPUT_JSON_FILE)
        print(f"Saved {len(results_data)} captions so far...")

    print(f"\n✅ Finished! Total successful captions: {len(results_data)}")
    print(f"Saved to {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main()
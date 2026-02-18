import os
import json
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from google import genai
from google.genai import types
from utils import image_from_filepath

# Using oauth2 config see https://ai.google.dev/palm_docs/oauth_quickstart
# Read default config from /home/your-user/.config/gcloud/application_default_credentials.json
# set env variable GOOGLE_APPLICATION_CREDENTIALS if necessary
client = genai.Client(
    vertexai=True,
    project="hybrid-cabinet-482822-p5",
    location="global"
) 


# --- Configuration ---
BASE_DIRECTORY = "/scratch/delineo_data/train/"
OUTPUT_FILE = "./ui_captions_dataset_v2.jsonl"

N_JOBS = -2
BATCH_SIZE = 100

# Set to an integer (e.g., 10) to test. 
# Set to None (or 0) to run the full dataset.
LIMIT = -1

SYSTEM_INSTRUCTION = """
You are an expert UI/UX designer creating training captions for a mobile UI generation model. 
Your task is to analyze the provided mobile screenshot and write a concise, professional prompt that a working product designer would naturally write to describe the interface. 

**CRITICAL ARCHITECTURE RULE:**
Your output MUST be strictly divided into two logical halves:
1.  **The Aesthetic Prefix (First ~20-40 words):** Global style, theme, color palette, and overall visual trend. 
2.  **The Component Body (Remaining words):** The layout and UI components described using standard industry terminology, natural phrasing, and key text values.

**Strict Output Rules:**
1.  **Start:** Always start with "High-fidelity mobile UI, [Theme/Vibe], [Color Palette]..." unless rejecting.
2.  **Length & Flow:** Be concise and natural (approx. 50-120 words max). Do not over-describe every pixel. Use professional shorthand and focus on the visual hierarchy.
3.  **Rejection (Quality Control):** If the image meets ANY of the following criteria, you must abort the description and return exactly and ONLY the phrase: "NOISY UI"
    * **Non-UI:** Blank screens, error messages, or purely text documents.
    * **Outdated UI:** Heavy skeuomorphism, archaic OS aesthetics (e.g., Windows 95) or outdated gradients.
    * **Amateur UI:** Broken layouts, bad padding/alignment, poor contrast, or chaotic hierarchy.

**Content Guidelines for Canny ControlNet:**
The Canny map will provide the spatial lines. Your prompt MUST provide the component names, fills, and key text.
* **Designer Vocabulary:** Use terms like: hero image, primary CTA, secondary button, bottom nav, FAB, modal, card component, list view, segmented control, whitespace, padding, elevation, drop shadow, opacity.
* **Colors & Materials:** Use clear, professional descriptors ("matte dark mode," "glassmorphism," "muted gray," "accent primary color").
* **Exact Text Rendering:** If prominent text exists (like headers or CTAs), enclose it in single quotes (e.g., CTA 'Sign Up').
* **Imagery:** Briefly state what the placeholder images represent (e.g., "avatar," "lifestyle hero image").

**Example (Simple UI):**
High-fidelity mobile UI, ultra-minimalist dark mode aesthetic, flat design, high contrast. A clean login screen with generous whitespace and a strong visual hierarchy. Center stage is a stark white text field for 'Email Address', sitting just above a pill-shaped primary CTA in electric lime reading 'Enter'. The top-center features a minimal geometric triangle logo. Flat elevation with no drop shadows.

**Example (Complex UI):**
High-fidelity mobile UI, modern travel aesthetic, airy lighting, glassmorphism elements, clean white background with soft yellow accents. A travel discovery feed featuring a transparent header overlapping a full-width nature hero image. Below is a frosted glass search bar with a 'Filter' icon. A horizontal scrollable section contains circular 'Story' avatars with gradient borders. The main vertical list view uses large, rounded-corner card components. A sample card shows a forest photo, bold serif title 'Alpine Lodge', a yellow star rating badge, and a bottom-right price tag '$120/night'. A minimalist line-art bottom nav includes Home, Search, Saved, Profile.
"""

generation_config = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=256,
    system_instruction=SYSTEM_INSTRUCTION,
    safety_settings=[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
    ]
)


def gather_all_images(base_dir):
    valid_paths = []
    print(f"Scanning {base_dir} for '_output.png' files...")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_output.png"):
                full_path = os.path.join(root, file)
                valid_paths.append(full_path)
    return valid_paths



def process_single_image(full_path):
    relative_path = os.path.relpath(full_path, BASE_DIRECTORY)
    image = image_from_filepath(full_path)
    if not image:
        print(f'Failed to load {relative_path}')
        return None

    for attempt in range(1, 9):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image], 
                config=generation_config
            )
            
            # Catch instances where safety settings block the response
            if not response.text:
                print(f"Empty response (Safety block?) for {relative_path}")
                return None
                
            caption = response.text.strip().replace("```", "").replace("\n", " ")
            return {"filename": relative_path, "caption": caption}
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                sleep_time = 10 + 2 ** attempt 
                time.sleep(sleep_time)
            else:
                print(f"Failed {relative_path} with error: {error_str}") 
                break
    
    print(f"Exhausted all retries for {relative_path} due to rate limits.")
    return None

def append_to_jsonl(data_list, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def load_existing_progress(filepath):
    processed = set()
    if not os.path.exists(filepath):
        return processed
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                processed.add(entry['filename'])
            except json.JSONDecodeError:
                continue
    return processed

def main():
    all_file_paths = gather_all_images(BASE_DIRECTORY)
    processed_files = load_existing_progress(OUTPUT_FILE)
    
    files_to_process = []
    for p in all_file_paths:
        rel_path = os.path.relpath(p, BASE_DIRECTORY)
        if rel_path not in processed_files:
            files_to_process.append(p)
    
    if LIMIT is not None and LIMIT > 0 and len(files_to_process) > LIMIT:
        print(f"\nLIMIT ACTIVE: Restricting run to first {LIMIT} images only.")
        files_to_process = files_to_process[:LIMIT]
    
    print(f"Total images found: {len(all_file_paths)}")
    print(f"Already done: {len(processed_files)}")
    print(f"To be processed: {len(files_to_process)}")
    print(f"Starting processing with {N_JOBS} threads...\n")

    if not files_to_process:
        print("✅ No new files to process!")
        return

    total_newly_processed = 0
    
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch_paths = files_to_process[i : i + BATCH_SIZE]
        
        batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(process_single_image)(p) for p in tqdm(batch_paths, desc=f"Batch {i//BATCH_SIZE + 1}", leave=False)
        )
        
        valid_results = [res for res in batch_results if res is not None]
        
        if valid_results:
            append_to_jsonl(valid_results, OUTPUT_FILE)
            total_newly_processed += len(valid_results)
            print(f"   Saved {len(valid_results)} new captions. (Total this run: {total_newly_processed})")

    print(f"\n✅ Finished! Added {total_newly_processed} new captions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
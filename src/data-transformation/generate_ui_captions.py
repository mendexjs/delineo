import os
import json
import time
from utils import crop_bars_from_filepath
from tqdm import tqdm
from joblib import Parallel, delayed
from google import genai
from google.genai import types

# Using oauth2 config see https://ai.google.dev/palm_docs/oauth_quickstart
# Read default config from /home/your-user/.config/gcloud/application_default_credentials.json
# set env variable GOOGLE_APPLICATION_CREDENTIALS if necessary
client = genai.Client(
    vertexai=True
) 


# --- Configuration ---
BASE_DIRECTORY = "/scratch/delineo_data/train/"
OUTPUT_FILE = "./ui_captions_dataset.jsonl"

N_JOBS = -1
BATCH_SIZE = 240

# Set to an integer (e.g., 10) to test. 
# Set to None (or 0) to run the full dataset.
LIMIT = -1

SYSTEM_INSTRUCTION = """
You are an expert UI/UX designer and prompt engineer creating training captions for a mobile UI generation model. 
Your task is to analyze the provided mobile screenshot and write a dense, visually descriptive caption that would allow a designer to recreate the interface exactly.

**Strict Output Rules:**
1.  **Format:** Return ONLY the raw text description. No Markdown, no quotes, no preambles.
2.  **Start:** Always start with "High-fidelity single screen mobile app UI design, no border, edge-to-edge view of..." unless it falls on following rejection case.
3.  **Adaptive Length:** * **Simple UI (e.g., Login, Search):** Keep it concise (40-80 words). Focus on typography, whitespace, and specific colors.
    * **Complex UI (e.g., Dashboard, Feed):** Go deep (100-160 words). Describe every section, card, and icon in detail.
4.  **Rejection:** If the image is blank, error, or purely text, return: "NOISY UI"

**Content Guidelines:**
1.  **Define the Screen:** "A minimalist login screen" vs "A data-dense analytics dashboard".
2.  **Layout & Geometry:** precise structures. "A centered card on a solid background" (Simple) vs "A masonry grid of 6 different sized cards" (Complex).
3.  **Visual Details:** Describe button shapes, shadow styles, and exact colors ("electric blue" not just "blue").
4.  **Content Hallucination:** Describe the *imagery* (photos, icons) vividly.
5.  **Typography & Style:** Mention font vibes ("bold sans-serif", "elegant serif") and aesthetic ("clean", "brutalist", "playful").

**Example (Simple UI):**
High-fidelity single screen mobile app UI design, no border, edge-to-edge view of a sign-in page. The background is a solid matte black. Center stage is a clean white input field with the placeholder text 'Email Address'. Below it, a bright 'electric lime' pill-shaped button with the text 'Enter'. The logo at the top is a minimal geometric triangle in white. The aesthetic is ultra-minimalist with ample negative space.

**Example (Complex UI):**
High-fidelity single screen mobile app UI design, no border, edge-to-edge view of a travel discovery feed. The header is transparent over a full-width hero photo of a mountain range. A floating glass-effect search bar sits at the top with a 'Filter' icon. Below, a horizontal scrolling list of circular 'Story' avatars with gradient rings. The main feed consists of large vertical cards, each displaying a high-res resort photo, a bold title 'Alpine Lodge', a star rating badge in yellow, and a price tag '$120/night' in the bottom right corner. The navigation bar at the bottom has four distinct icons: Home, Search, Saved, Profile.
"""

generation_config = types.GenerateContentConfig(
    temperature=0.4,
    max_output_tokens=200,
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
    print(f"📂 Scanning {base_dir} for '_output.png' files...")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_output.png"):
                full_path = os.path.join(root, file)
                valid_paths.append(full_path)
    return valid_paths



def process_single_image(full_path):
    relative_path = os.path.relpath(full_path, BASE_DIRECTORY)
    cropped_image = crop_bars_from_filepath(full_path)
    if not cropped_image:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[cropped_image], 
                config=generation_config
            )
            
            caption = response.text.strip().replace("```", "").replace("\n", " ")
            return {"filename": relative_path, "caption": caption}
            
        except Exception as e:
            # Simple error handling for rate limits or server errors
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"Failed {relative_path}: {e}") # Uncomment to debug
                break
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
    
    if LIMIT and len(files_to_process) > LIMIT:
        print(f"\n⚠️ LIMIT ACTIVE: Restricting run to first {LIMIT} images only.")
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
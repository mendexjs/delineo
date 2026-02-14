import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from google import genai
from google.genai import types
from utils import image_from_filepath

# Using oauth2 config see https://ai.google.dev/palm_docs/oauth_quickstart
client = genai.Client(
    vertexai=True,
    project="hybrid-cabinet-482822-p5",
    location="global"
) 

# --- Configuration ---
# Point this to the root of your dataset folders
BASE_DIRECTORY = "/home/matheus_mendes/projects/delineo/src/training/lora/dataset/"

N_JOBS = 10
BATCH_SIZE = 10
LIMIT = -1

# Maintained your strict rules for natural designer vocabulary, rejection, and structure
SYSTEM_INSTRUCTION = """
You are an expert UI/UX designer creating training captions for a mobile UI generation model. 
Your task is to analyze the provided mobile screenshot and write a concise, professional prompt that a working product designer would naturally write to describe the interface. 

**CRITICAL ARCHITECTURE RULE:**
Your output MUST be strictly divided into two logical halves:
1.  **The Aesthetic Prefix (First ~20-40 words):** Global style, theme, color palette, and overall visual trend. 
2.  **The Component Body (Remaining words):** The layout and UI components described using standard industry terminology, natural phrasing, and key text values. Avoid making the description super detailed; keep it matched to how a designer would naturally describe it on a daily basis.

**Strict Output Rules:**
1.  **Start:** Always start with "High-fidelity mobile UI, [Theme/Vibe], [Color Palette]...".
2.  **Length & Flow:** Be concise and natural (approx. 50-120 words max). Do not over-describe every pixel. Use professional shorthand and focus on the visual hierarchy.

**Content Guidelines for Canny ControlNet:**
The Canny map will provide the spatial lines. Your prompt MUST provide the component names, fills, and key text.
* **Designer Vocabulary:** Use terms like: hero image, primary CTA, secondary button, bottom nav, FAB, modal, card component, list view, segmented control, whitespace, padding, elevation, drop shadow, opacity.
* **Colors & Materials:** Use clear, professional descriptors ("matte dark mode," "glassmorphism," "muted gray," "accent primary color").
* **Exact Text Rendering:** If prominent text exists (like headers or CTAs), enclose it in single quotes (e.g., CTA 'Sign Up').
* **Imagery:** Briefly state what the placeholder images represent (e.g., "avatar," "lifestyle hero image").
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

def gather_dataset_files(base_dir):
    """
    Scans for images and determines their corresponding .txt file path.
    Returns a list of dicts with paths and folder context.
    """
    files_to_process = []
    print(f"Scanning {base_dir} for images...")
    
    for root, dirs, files in os.walk(base_dir):
        # Extract the folder name (e.g., 'shop', 'finances') for context
        category = os.path.basename(root)
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext == '.png':
                img_path = os.path.join(root, file)
                # Define the target .txt file alongside the image
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                
                # Only add if the .txt file doesn't already exist
                if not os.path.exists(txt_path):
                    files_to_process.append({
                        "img_path": img_path,
                        "txt_path": txt_path,
                        "category": category
                    })
                    
    return files_to_process


def process_single_image(file_data):
    img_path = file_data["img_path"]
    txt_path = file_data["txt_path"]
    category = file_data["category"]
    
    image = image_from_filepath(img_path)
    if not image:
        print(f"Failed to load {img_path}")
        return False

    # Inject the folder category as a hidden prompt to guide the VLM
    context_prompt = f"Context: This UI is from the '{category}' category. Ensure the terminology matches this context."

    for attempt in range(1, 7):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image, context_prompt], 
                config=generation_config
            )
            
            if not response.text:
                print(f"Empty response for {img_path}")
                return False
                
            caption = response.text.strip().replace("```", "").replace("\n", " ")
                
            # Write directly to the .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
                
            return True
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str or "quota" in error_str.lower():
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                print(f"Failed {img_path} with error: {error_str}") 
                break
                
    print(f"Exhausted all retries for {img_path} due to rate limits.")
    return False


def main():
    files_to_process = gather_dataset_files(BASE_DIRECTORY)
    
    if LIMIT is not None and LIMIT > 0 and len(files_to_process) > LIMIT:
        print(f"\nLIMIT ACTIVE: Restricting run to first {LIMIT} images only.")
        files_to_process = files_to_process[:LIMIT]
    
    print(f"Images needing captions: {len(files_to_process)}")
    
    if not files_to_process:
        print("No new files to process! All images have matching .txt files.")
        return

    print(f"Starting processing with {N_JOBS} threads...\n")
    total_newly_processed = 0
    
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch_data = files_to_process[i : i + BATCH_SIZE]
        
        batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(process_single_image)(data) for data in tqdm(batch_data, desc=f"Batch {i//BATCH_SIZE + 1}", leave=False)
        )
        
        # Count how many returned True (successfully saved .txt)
        successes = sum(1 for res in batch_results if res)
        total_newly_processed += successes
        print(f"   Saved {successes} new .txt captions. (Total this run: {total_newly_processed})")

    print(f"\n✅ Finished! Added {total_newly_processed} new .txt caption files to your dataset directories.")

if __name__ == "__main__":
    main()
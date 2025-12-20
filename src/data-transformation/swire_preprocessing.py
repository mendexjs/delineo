import os
import cv2
import glob
from pathlib import Path
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
N_JOBS = -1 # All CPUs available


"""
TODO: Parse Rico annotation.json and avoid samples with forbidden components e.g. Web View
"""


# These samples are corrupted
SWIRE_INVALID_SAMPLES = (
    '71922_2.jpg',
    '567986_2.jpg',
    '33181_1.jpg',
    '30635_1.jgp',
    '29822_1.jpg',
    '29799_1.jpg',
    '29770_1.jpg',
    '29766_1.jpg',
    '23424_1.jpg',
    '23426_1.jpg',
    '23427_1.jpg',
    '20613_1.jpg',
    '8946_3.jpg',
    '2961_3.jpg',
    '2959_1.jpg',
    '2220_2.jpg',
)

SWIRE_VALIDATION_SAMPLES = (
    '4240', # High-fidelity mobile UI design of the H&M app main page. Features a hero carousel with fashion trends, navigation for 'Women' and 'Men' categories, and a bright promotional banner displaying 'Last Day of Festival Shop - 50% OFF'. Minimalist, modern e-commerce aesthetic.
    '12922', # High-fidelity mobile UI design of a vacation rental app search feed. Features a vertical list of property cards with large high-quality room photography, description, price per night, and star rating. Includes a search bar with 'Anywhere' and 'Anytime' generic filters at the top.
    '33178', # High-fidelity mobile UI design of an interest selection screen. Features a vertical list of topic rows including 'Alcoholic Drinks', 'Animals', 'Arts', 'Beauty' and 'Beer'. Each row contains a small square thumbnail image, category title, subtle follower count text, and a selection checkbox on the right. Clean, modern list interface.
    '56563', # High-fidelity mobile UI design of a local deals app feed. Features a top search bar and horizontal category filters for 'Goods', 'Things to do', 'Beauty', and 'Restaurants'. Below is a list of offer cards displaying vibrant photos, deal descriptions, location pins, and discounted prices. Vibrant, promotional marketplace interface.
)

# mud
# '408' High-fidelity mobile UI design of a registration screen. Features a prominent 'EasyBox' logo in blue and orange at the top. Below is a stacked form with input fields for Name, Email, Password, and Confirm Password. Includes a Terms of Use agreement checkbox and a solid blue primary submit button. Clean, trustworthy aesthetic with blue accents.
# '14976 High-fidelity mobile UI design of a modern real estate search screen. Features a bright, high-quality hero background photo of a sunny residential landscape. Overlaid centrally are buttons to chose 'For Sale' or 'For Rent', above a prominent search bar with placeholder text 'Location or Address', and quick-action buttons for 'Search by commute' and 'Search nearby'. Clean, translucid interface elements.
def setup_paths():
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent
    
    # Input Directories
    swire_dir = src_dir / "raw-data" / "swire"
    rico_dir = src_dir / "raw-data" / "rico"
    
    # Output to /scratch -- NVMe SSD
    out_train_dir = Path("/scratch/delineo_data/train/swire")
    out_validation_dir = Path("/scratch/delineo_data/validation")
    
    return swire_dir, rico_dir, out_train_dir, out_validation_dir

def process_single_pair(swire_path, rico_dir, out_train_dir, out_validation_dir):
    """
    Worker function to process a single image pair.
    """
    try:
        filename = swire_path.name
        rico_id = filename.split("_")[0]

        if filename in SWIRE_INVALID_SAMPLES:
            return f"SKIP: {rico_id} is listed in SWIRE_INVALID_SAMPLES"
        
        # 2. Find Correspondence in Rico
        rico_path = None
        potential_path = rico_dir / f"{rico_id}.jpg"
        if potential_path.exists():
            rico_path = potential_path
        
        if rico_path is None:
            return f"SKIP: No match for ID {rico_id}"

        # 3. Load Images
        img_swire = cv2.imread(str(swire_path))
        img_rico = cv2.imread(str(rico_path))

        if img_swire is None or img_rico is None:
            return f"ERROR: Failed to load images for {rico_id}"

        # 4. Resize
        # SWIRE (Wireframe) -> Use NEAREST to keep sharp edges (black/white)
        swire_resized = cv2.resize(img_swire, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # RICO (Screenshot) -> Use AREA for high-quality downsampling of UI text
        rico_resized = cv2.resize(img_rico, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

       # Convert BGR (OpenCV) to RGB (PIL)
        swire_pil = Image.fromarray(cv2.cvtColor(swire_resized, cv2.COLOR_BGR2RGB))
        rico_pil = Image.fromarray(cv2.cvtColor(rico_resized, cv2.COLOR_BGR2RGB))
        
        # Save
        export_dir = out_validation_dir if rico_id in SWIRE_VALIDATION_SAMPLES else out_train_dir
        swire_pil.save(export_dir / f"{swire_path.stem}_input.png", format="PNG", compress_level=1)
        rico_pil.save(export_dir / f"{rico_id}_output.png", format="PNG", compress_level=1)

        return None

    except Exception as e:
        return f"EXCEPTION: {filename} - {str(e)}"

def main():
    # 1. Setup Directories
    swire_dir, rico_dir, out_train_dir, out_validation_dir = setup_paths()
    
    print(f"--- Configuration ---")
    print(f"Input Swire: {swire_dir}")
    print(f"Input Rico:  {rico_dir}")
    print(f"Output (train):      {out_train_dir}")
    print(f"Output (validation):      {out_validation_dir}")
    print(f"Target Res:  {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"---------------------")

    # Validate Inputs
    if not swire_dir.exists():
        print(f"❌ Error: Swire directory not found at {swire_dir}")
        return
    if not rico_dir.exists():
        print(f"❌ Error: Rico directory not found at {rico_dir}")
        return

    # Create Output Dirs
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_validation_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect Files
    # Get all .jpg files in swire directory
    swire_files = list(swire_dir.glob("*.jpg"))
    total_files = len(swire_files)
    print(f"Found {total_files} swire candidates. Processing...")

    # 3. Parallel Processing
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_pair)(
            p, rico_dir, out_train_dir, out_validation_dir
        ) for p in tqdm(swire_files, total=total_files, unit="img")
    )

    # 4. Reporting
    skipped = [r for r in results if r and r.startswith("SKIP")]
    errors = [r for r in results if r and (r.startswith("ERROR") or r.startswith("EXCEPTION"))]
    success_count = len(results) - len(skipped) - len(errors)

    print(f"\n--- Processing Complete ---")
    print(f"✅ Successfully processed: {success_count}")
    print(f"⏭️  Skipped (No match):    {len(skipped)}")
    print(f"❌ Errors:                 {len(errors)}")
    
    if errors:
        print("\nFirst 10 errors:")
        for e in errors[:10]:
            print(e)

if __name__ == "__main__":
    main()
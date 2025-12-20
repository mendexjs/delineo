accelerate launch train_controlnet_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
  --controlnet_model_name_or_path="stabilityai/stable-diffusion-3.5-large-controlnet-canny" \
  --output_dir="./sd35-delineo-finetuned-v1" \
  --cache_dir=$HF_HOME \
  --train_data_dir="/scratch/delineo_data/train" \
  --conditioning_image_column="input" \
  --image_column="output" \
  --caption_column="text" \
  --resolution=1024 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --num_train_epochs=20 \
  --checkpoints_total_limit=5 \
  --checkpointing_steps=1250 \
  --mixed_precision="bf16" \
  --report_to="tensorboard" \
  --validation_steps=1250 \
  --validation_image \
    "/scratch/delineo_data/validation/408_input.png" \
    "/scratch/delineo_data/validation/14976_input.png" \
    "/scratch/delineo_data/validation/4240_1_input.png" \
    "/scratch/delineo_data/validation/12922_2_input.png" \
    "/scratch/delineo_data/validation/33178_1_input.png" \
    "/scratch/delineo_data/validation/56563_3_input.png" \
  --validation_prompt \
    "High-fidelity mobile UI design of a registration screen. Features a prominent 'EasyBox' logo in blue and orange at the top. Below is a stacked form with input fields for Name, Email, Password, and Confirm Password. Includes a Terms of Use agreement checkbox and a solid blue primary submit button. Clean, trustworthy aesthetic with blue accents." \
    "High-fidelity mobile UI design of a modern real estate search screen. Features a bright, high-quality hero background photo of a sunny residential landscape. Overlaid centrally are buttons to chose 'For Sale' or 'For Rent', above a prominent search bar with placeholder text 'Location or Address', and quick-action buttons for 'Search by commute' and 'Search nearby'. Clean, translucid interface elements." \
    "High-fidelity mobile UI design of the H&M app main page. Features a hero carousel with fashion trends, navigation for 'Women' and 'Men' categories, and a bright promotional banner displaying 'Last Day of Festival Shop - 50% OFF'. Minimalist, modern e-commerce aesthetic." \
    "High-fidelity mobile UI design of a vacation rental app search feed. Features a vertical list of property cards with large high-quality room photography, description, price per night, and star rating. Includes a search bar with 'Anywhere' and 'Anytime' generic filters at the top." \
    "High-fidelity mobile UI design of an interest selection screen. Features a vertical list of topic rows including 'Alcoholic Drinks', 'Animals', 'Arts', 'Beauty' and 'Beer'. Each row contains a small square thumbnail image, category title, subtle follower count text, and a selection checkbox on the right. Clean, modern list interface." \
    "High-fidelity mobile UI design of a local deals app feed. Features a top search bar and horizontal category filters for 'Goods', 'Things to do', 'Beauty', and 'Restaurants'. Below is a list of offer cards displaying vibrant photos, deal descriptions, location pins, and discounted prices. Vibrant, promotional marketplace interface."
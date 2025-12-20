To download the models, ensure to set your hugging face token
```bash
export HF_TOKEN"hf-xyz"
```

Downloading the StableDiffusion 3.5 + controlnet from terminal using python:
```bash
python
from huggingface_hub import hf_hub_download
hf_hub_download("stabilityai/stable-diffusion-3.5-large", "sd3.5_large.safetensors", local_dir="/scratch/models")
hf_hub_download("stabilityai/stable-diffusion-3.5-large", "text_encoders/clip_g.safetensors", local_dir="/scratch/models")
hf_hub_download("stabilityai/stable-diffusion-3.5-large", "text_encoders/clip_l.safetensors", local_dir="/scratch/models")
hf_hub_download("stabilityai/stable-diffusion-3.5-large", "text_encoders/t5xxl_fp16.safetensors", local_dir="/scratch/models")
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors", local_dir="/scratch/models")
exit()
# moving files and renaming to what the python inference script expects to read
mv /scratch/models/text_encoders/* /scratch/models && mv /scratch/models/t5xxl_fp16.safetensors /scratch/models/t5xxl.safetensors && rm -r /scratch/models/text_encoders
```

## Testing SD3.5
```bash
cd src/training/sd3.5-main
python3 sd3_infer.py --prompt "cute wallpaper art of a cat holding a rose"
```
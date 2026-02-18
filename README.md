# Delineo
GenAI pipeline to transform UI hand sketches with prompt in high level mockups via open-source diffusion model's control techniques.

### Conda set up for ControlNet

Edit `delineo-environment.yaml` and update `prefix` with your conda home path.
If you plan to train the ControlNet, you also need to change the `HF_TOKEN` variable with your hugging face token.

Create env from file: `conda env create -f delineo-environment.yml`
Export env to a file: `conda export > delineo-environment.yaml`

Activate the environment to run following steps:
`conda activate delineo`

---

# Training Setup

## Datasets and data sources

### VINS Dataset
>**Description:** VINS is a recent UI dataset with xml files containing bounding boxes for components, annotated by humans. <br/>
**Usage:** We use VINS to programmatically create 3,119 synthetic sketches samples, transforming components declared in the xml file to shapes and adding noise to emulate human hand sketches.<br/><br/>
**Paper:** https://dl.acm.org/doi/10.1145/3411764.3445762 <br/>
**Repo:** https://github.com/sbunian/VINS <br/>
**Download Link:** https://drive.google.com/file/d/1cbTRT3ky_zTveb_NnNPLUO9ZGXHvCcUl/view?usp=sharing

Download via cli:
```bash
mkdir -p src/raw-data/vins && ZIP_PATH=./src/raw-data/vins/vins_dataset.zip && gdown 1ucti2ujNJgKV-Ts_mXALA7BZ97iPv8h6 -O ${ZIP_PATH} && unzip -q -o ${ZIP_PATH} -d src/raw-data && rm -r ./src/raw-data/vins && mv src/raw-data/All\ Dataset src/raw-data/vins
```

### MUD Dataset
>**Description:** MUD is a UI dataset intended to improve view hierarchy issues that Rico has. It's more recent, so it also has more modern app user interfaces. <br/>
**Usage:** We use MUD to programmatically create 7,000 synthetic sketches samples, transforming components declared in the json file to shapes and adding noise to emulate human hand sketches.<br/><br/>
**Paper:** https://dl.acm.org/doi/10.1145/3613904.3642350 <br/>
**Repo:** https://github.com/sidongfeng/MUD/tree/main <br/>
**Download Link:** https://drive.google.com/file/d/1cbTRT3ky_zTveb_NnNPLUO9ZGXHvCcUl/view?usp=sharing

Download via cli:
```bash
mkdir -p src/raw-data/mud && TAR_PATH=./src/raw-data/mud/mud_dataset.tar.gz && gdown 1cbTRT3ky_zTveb_NnNPLUO9ZGXHvCcUl -O ${TAR_PATH} && tar -xzf ${TAR_PATH} -C src/raw-data/mud --strip-components=1 && rm ${TAR_PATH} && rm src/raw-data/mud/._*
```

# Swire Dataset
>**Description:** Swire has sketches created by real designers, and files are enumerated according to Rico dataset IDs. The sullfix represent an anonymous designer. e.g. 123_1.jpg -- Rico UI 123, designer codename 1.<br/>
**Usage:** This is our gold-dataset, containing real domain examples of how designers sketch UIs. We'll combine this dataset with out mud-synthetic-sketches to train the diffusion control.<br/><br/>
**Paper:** https://dl.acm.org/doi/10.1145/3290605.3300334 <br/>
**Repo:** https://github.com/huang4fstudio/swire <br/>
**Download Link:** https://storage.googleapis.com/crowdstf-rico-uiuc-4540/swire_dataset_v0.1/sketches.zip

Download via cli:
```bash
mkdir -p src/raw-data && ZIP_PATH=./src/raw-data/swire_dataset.zip && wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/swire_dataset_v0.1/sketches.zip -O ${ZIP_PATH} && unzip -q -o ${ZIP_PATH} -d src/raw-data  && rm ${ZIP_PATH} && mv src/raw-data/sketches src/raw-data/swire
```

# Rico Dataset
>**Description:** Rico has over 60K UIs gathered from app stores, with side modules containing view hierarchies, apps metadata, and annotations.<br/>
**Usage:** We use Rico only to get the original UI image that will be referenced as target for a given Swire sketch.<br/><br/>
**Paper:** https://dl.acm.org/doi/10.1145/3126594.3126651 <br/>
**Website:** https://www.interactionmining.org/archive/rico
**Download Link:** https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz

Download via cli:
```bash
mkdir -p src/raw-data/rico && TAR_PATH=./src/raw-data/rico/rico_dataset.tar.gz && wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz -O ${TAR_PATH} && tar -xzf ${TAR_PATH} -C src/raw-data/rico --strip-components=1 && rm ${TAR_PATH} && cd src/raw-data/rico && rm *.json
```

---

### Preprocessing datasets for training

> Alway ensure to edit the script output path to the location you like to output the data 

```bash
cd src/data-transformation && \
python mud_preprocessing.py && \
python vins_preprocessing.py && \
python swire_preprocessing.py
```

### Generate captions with Gemini (optional)

The captions I generated are available in `src/data-transformation/ui_captions_dataset_v2.jsonl`. You can use those unless you want to rework the prompt and regenerate.

I spent about 7 USD to generate the captions, google cloud gets you 300 USD credits for new accounts.
Ensure to setup your credentials and update projectId in `src/data-transformation/generate_ui_captions.py`
Get your application_default_credentials.json (Docs: https://ai.google.dev/palm_docs/oauth_quickstart)

> Note: The Gemini API is subject to rate limit. Sometimes the script will finish but you still need to run again the next day to generate captions for the ones it failed.
> There is retry logic implemented, I had to run it about 4 times to get all captions, tough. 

```bash
cd src/data-transformation && \
python generate_ui_captions.py
```

### Generating training metadata

Finally, we can generate the metadata.jsonl, joining images, sketch and captions.

```bash
cd src/data-transformation && \
python prepare_training_metadata.py
```

---

### Training ControlNet

The ControlNet training was done in a single NVIDIA Tesla A100 80gb. It took about 5 days to complete training.
You might need install and use `tmux` to allow you have persistent terminal sessions in ssh and avoid training interruptions.

```bash
conda activate delineo

cd src/training/controlnet && /
bash start_training.sh
```

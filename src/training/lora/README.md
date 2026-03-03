### Setup

We use diffusers lora training script for SD 3.5

```bash
cd src/training/lora && git clone git@github.com:huggingface/diffusers.git@f8d3db9ca7dc4bd15405269ae5d2744d0eb72658
```

Edit `lora-environment.yaml` and update `prefix` with your conda home path, and `HF_TOKEN` with your hugging face token

```bash
conda env create -f lora_env.yml
conda activate lora
python preprocess_dataset.py && python make_metadata.py
```

If you run in issues creating the environment from the yaml, follow the instructions in https://github.com/huggingface/diffusers/blob/f8d3db9ca7dc4bd15405269ae5d2744d0eb72658/examples/dreambooth/README_sd3.md

### Training
```bash
bash start_lora_training.sh
```

We tested different rank, learning rate and bach sizes. The preferred version is the checkpoint of 1000 steps: rank 64, lr 1e-4, encoder lr 1e-5.
---

### Epochs evaluation

We have a script that runs inference with fixed seeds across all checkpoints to we can see how the outputs evolves and find the best checkpoint.

We generate a batch of 5 images per validation pair, for 5 different samples across all epochs.
The output is one image grid per sample, showing evolution across all checkpoints. View an example in `src/experiments/lora/epochs_evolution`.
The script takes about 2 hours to run on an A100 80GB. You can adjust `PROMPT_BATCH_SIZE` according to your GPU resources.
You can also add more validation examples or comment out to speed up the script.
It's also possible to enable cpu offload to save VRAM by slowing down inference. Add `pipe.enable_model_cpu_offload()` after pipe instantiation.

```bash
cd ../../experiments && \
python epochs_evaluation_script.py
```

---

### High quality UIs dataset for LoRA training

The LoRA dataset was manually curated from modern and clean mobile UI extracted from Figma free community design resources (https://www.figma.com/community/mobile-apps/ui?resource_type=mixed&editor_type=figma&price=free&sort_by=all_time)

Sources:
  - https://www.figma.com/community/file/1322236579213422290
  - https://www.figma.com/community/file/1143575071825582037
  - https://www.figma.com/community/file/882645007956337261
  - https://www.figma.com/community/file/882888114457713282
  - https://www.figma.com/community/file/1106038596434269509
  - https://www.figma.com/community/file/1091615514005406765
  - https://www.figma.com/community/file/1169928945460966636
  - https://www.figma.com/community/file/1169625825293818878
  - https://www.figma.com/community/file/1116625179283253250
  - https://www.figma.com/community/file/1169922911743790308
  - https://www.figma.com/community/file/1169924584217287910
  - https://www.figma.com/community/file/1394518755367476147
  - https://www.figma.com/community/file/1134379406586739829
  - https://www.figma.com/community/file/885503108183774962
  - https://www.figma.com/community/file/1143115506742537849
  - https://www.figma.com/community/file/1323887283985127984
  - https://www.figma.com/community/file/1172153496393189176
  - https://www.figma.com/community/file/1235854850814701017
  - https://www.figma.com/community/file/1377526482587017983
  - https://www.figma.com/community/file/1043459494713403436
  - https://www.figma.com/community/file/1283671441735791435
  - https://www.figma.com/community/file/1116708627748807811
  - https://www.figma.com/community/file/1370757927948360864
  - https://www.figma.com/community/file/1264098337558102933
  - https://www.figma.com/community/file/1340685192199391354
  - https://www.figma.com/community/file/1136591795316360905
  - https://www.figma.com/community/file/1428695240766387403
  - https://www.figma.com/community/file/1249743721562101377
  - https://www.figma.com/community/file/1321464360558173342
  - https://www.figma.com/community/file/1168885651980114323
  - https://www.figma.com/community/file/1343984207344248339
  - https://www.figma.com/community/file/1146678238901785717
  - https://www.figma.com/community/file/1362309395455453748
  - https://www.figma.com/community/file/1283725118401003453
import os

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

from transformers import AutoProcessor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

max_length = 350
data_dir = "/scratch/delineo_data/train"
out_dir = "/scratch/delineo_outputs/blip2_ui_lora_v2"
dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
print(dataset.features)


dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]


# 2) Dataset wrapper (same vibe as HF example)
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, image_column="output", text_column="text"):
        self.dataset = dataset
        self.processor = processor
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # item["output"] is already decoded as a PIL image by datasets.Image
        encoding = self.processor(images=item[self.image_column], return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}  # remove batch dim
        encoding["text"] = item[self.text_column]
        return encoding


def collate_fn(batch, processor, max_length=max_length):
    pixel_values = torch.stack([ex["pixel_values"] for ex in batch])

    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = [PROMPT + ex["text"] for ex in batch]
    text_inputs = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100   # ignore PADs in loss

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # simplest + usually correct for OPT blocks
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# 4) Dataloaders
train_dataset = ImageCaptioningDataset(train_ds, processor, image_column="output", text_column="text")
val_dataset   = ImageCaptioningDataset(val_ds, processor, image_column="output", text_column="text")

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=8,
    collate_fn=lambda b: collate_fn(b, processor, max_length=max_length),
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=8,
    collate_fn=lambda b: collate_fn(b, processor, max_length=max_length),
)
PROMPT = "Describe this mobile UI screen in detail: "

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
grad_accum_steps = 4
num_epochs = 6
num_training_steps = num_epochs * len(train_loader) // grad_accum_steps
num_warmup_steps = int(0.03 * num_training_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()



@torch.no_grad()
def sample_generations(model, processor, val_ds, n=3, max_new_tokens=250, max_length=max_length, dtype=torch.bfloat16, mask_prompt=True):
    model.eval()
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompt_ids = tok(PROMPT, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    for i in range(n):
        ex = val_ds[i]
        img = ex["output"]
        gt_text = ex["text"]

        # ---------- Teacher-forced loss (same objective as training) ----------
        full_text = PROMPT + gt_text
        text_inputs = tok(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(model.device)
        attn = text_inputs["attention_mask"].to(model.device)

        labels = input_ids.clone()
        labels[attn == 0] = -100           # ignore PAD
        if mask_prompt:
            labels[:, :prompt_len] = -100  # ignore prompt tokens

        vision = processor(images=img, return_tensors="pt")
        pixel_values = vision["pixel_values"].to(model.device, dtype)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = out.loss.item()

        # ---------- Generation ----------
        gen_inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            min_new_tokens=200,        # helps avoid super short captions
            length_penalty=1.2,       # encourages longer outputs with beams
        )
        pred = tok.decode(gen_ids[0], skip_special_tokens=True)

        print("\n---")
        print(f"loss: {loss:.4f}")
        print("GT:", gt_text[:300])
        print("PR:", pred[:300])

    model.train()
for epoch in tqdm(range(num_epochs), desc="Training Blip"):
    print("Epoch:", epoch)
    for step, batch in enumerate(tqdm(train_loader, desc="Steps", total=len(train_loader))):
        pixel_values = batch["pixel_values"].to(model.device, torch.bfloat16)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=batch["labels"].to(device),
        )

        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    sample_generations(model, processor, val_ds, n=3)
    path = os.path.join(out_dir, f"epoch-{epoch}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"Saved checkpoint: {path}")

model.save_pretrained(out_dir)
processor.save_pretrained(out_dir)
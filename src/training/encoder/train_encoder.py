import random
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import json
import torch
import torch.nn.functional as F
import pandas as pd
import open_clip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm



# --- Configurações Otimizadas ---
BASE_DIR = Path("/scratch/delineo_data/train")
JSONL_PATH = BASE_DIR / "metadata.jsonl"
BATCH_SIZE = 48 
EPOCHS = 15 # Aumentado para melhor convergência com LR baixa
LEARNING_RATE = 1e-6 # Reduzida para ajuste fino de alta qualidade
DEVICE = torch.device("mps")
OUTPUT_DIR = Path("/scratch/delineo_outputs/encoder")
OUTPUT_FILE = OUTPUT_DIR / "ui_semantic_encoder_L14.pth"
LOG_FILE = OUTPUT_DIR / "training_log.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class UIContrastiveDataset(Dataset):
    def __init__(self, samples, base_dir, desc="Dataset"):
        self.base_dir = Path(base_dir)
        self.samples = samples
        
        # Pre-processamento básico para tirar o peso da CPU no loop
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        
        print(f"🚀 Carregando e Redimensionando {len(self.samples)} imagens ({desc}) na RAM...")
        self.cached_sketches = []
        self.cached_uis = []
        
        for s in tqdm(self.samples, desc=f"Caching {desc}"):
            # Carrega e já redimensiona (operação mais lenta da PIL)
            sk = self.resize(Image.open(self.base_dir / s['input_file_name']).convert('RGB'))
            ui = self.resize(Image.open(self.base_dir / s['output_file_name']).convert('RGB'))
            
            self.cached_sketches.append(sk)
            self.cached_uis.append(ui)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sk_img = self.cached_sketches[idx]
        ui_img = self.cached_uis[idx]
        
        # Aplica apenas transformações rápidas de tensor no getitem
        # Se quiser Augmentation, adicione aqui (ex: RandomAffine)
        sketch = self.normalize(self.to_tensor(sk_img))
        ui = self.normalize(self.to_tensor(ui_img))
        
        return sketch, ui

def calculate_accuracy(logits):
    labels = torch.arange(len(logits)).to(logits.device)
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def calculate_topk_accuracy(logits, k=5):
    batch_size = logits.size(0)
    actual_k = min(k, batch_size)
    labels = torch.arange(batch_size).to(logits.device)
    _, topk_indices = logits.topk(actual_k, dim=1)
    correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
    return correct.any(dim=1).float().mean().item()

def validate(model, loader):
    model.eval()
    val_loss, val_acc1, val_acc5 = 0, 0, 0
    with torch.no_grad():
        for sketches, uis in loader:
            sketches, uis = sketches.to(DEVICE), uis.to(DEVICE)
            
            # Autocast também na validação para manter consistência e velocidade
            with torch.amp.autocast(device_type='mps', enabled=True):
                s_feat = model.encode_image(sketches)
                u_feat = model.encode_image(uis)
                s_feat /= s_feat.norm(dim=-1, keepdim=True)
                u_feat /= u_feat.norm(dim=-1, keepdim=True)
                logits = (s_feat @ u_feat.T) / 0.07
                loss = F.cross_entropy(logits, torch.arange(len(logits)).to(DEVICE))
            
            val_loss += loss.item()
            val_acc1 += calculate_accuracy(logits)
            val_acc5 += calculate_topk_accuracy(logits, k=5)
    return val_loss / len(loader), val_acc1 / len(loader), val_acc5 / len(loader)

def train():
    # 1. Carregamento e Split
    all_metadata = []
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['input_file_name'].startswith(('swire/', 'vins/')):
                all_metadata.append(data)
    
    random.seed(42)
    random.shuffle(all_metadata)
    val_samples = all_metadata[:500]
    train_samples = all_metadata[500:]
    
    # 2. Datasets e Loaders (num_workers=0 é crucial aqui)
    train_ds = UIContrastiveDataset(train_samples, BASE_DIR, desc="Treino")
    val_ds = UIContrastiveDataset(val_samples, BASE_DIR, desc="Validação")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Modelo ViT-L-14
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.to(DEVICE)
    
    for param in model.parameters(): param.requires_grad = False
    for param in model.visual.transformer.resblocks[-12:].parameters(): param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_val_acc1 = 0.0

    import gc # Import no topo do arquivo

# ... dentro da função train() ...

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc1, train_acc5 = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for sketches, uis in pbar:
            # Garante que os dados entrem no DEVICE (MPS)
            sketches, uis = sketches.to(DEVICE), uis.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            s_feat = model.encode_image(sketches)
            u_feat = model.encode_image(uis)
            
            # Estabilidade numérica
            s_feat = s_feat / (s_feat.norm(dim=-1, keepdim=True) + 1e-6)
            u_feat = u_feat / (u_feat.norm(dim=-1, keepdim=True) + 1e-6)
            
            logits = (s_feat @ u_feat.T) / 0.07
            loss = F.cross_entropy(logits, torch.arange(len(logits)).to(DEVICE))

            # Backward pass e Clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Métricas - Extraindo apenas valores escalares (.item())
            current_loss = loss.item()
            acc1 = calculate_accuracy(logits)
            acc5 = calculate_topk_accuracy(logits, k=5)
            
            train_loss += current_loss
            train_acc1 += acc1
            train_acc5 += acc5
            
            pbar.set_postfix({
                "Loss": f"{current_loss:.3f}", 
                "T1": f"{acc1:.2f}",
                "T5": f"{acc5:.2f}"
            })
        
        # --- LIMPEZA DE MEMÓRIA PÓS-TREINO ---
        scheduler.step()
        
        # Validação
        v_loss, v_acc1, v_acc5 = validate(model, val_loader)

        print(f"\n📊 Época {epoch+1}:")
        print(f"   Train T1: {train_acc1/len(train_loader):.4f} | Val T1: {v_acc1:.4f}")
        print(f"   Train T5: {train_acc5/len(train_loader):.4f} | Val T5: {v_acc5:.4f}")
        print(f"   Loss {v_loss:.4f}")
        
        # Salva no histórico garantindo que são apenas tipos nativos do Python
        history.append({
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss/len(train_loader)),
            "train_acc1": float(train_acc1/len(train_loader)),
            "train_acc5": float(train_acc5/len(train_loader)),
            "val_loss": float(v_loss),
            "val_acc1": float(v_acc1),
            "val_acc5": float(v_acc5)
        })

        pd.DataFrame(history).to_csv(LOG_FILE, index=False)

        if v_acc1 > best_val_acc1:
            best_val_acc1 = v_acc1
            torch.save(model.state_dict(), OUTPUT_FILE)
            print(f"⭐ Melhor modelo salvo!")

        # --- GESTÃO DE RECURSOS DO MPS ---
        # 1. Limpa o cache de memória do Metal (MPS)
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        
        # 2. Força o Garbage Collector do Python
        gc.collect()

if __name__ == "__main__":
    train()
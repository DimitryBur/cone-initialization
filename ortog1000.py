import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json, gc

class OrthoMLPLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        # Ортогональная инициализация (с коэффициентом для ReLU)
        nn.init.orthogonal_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return self.relu(self.ln(self.linear(x)))

# [Функции get_eff_rank, get_data, get_current_ranks идентичны скрипту выше]
def get_eff_rank(tensor):
    with torch.no_grad():
        s = torch.linalg.svdvals(tensor)
        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        return torch.exp(entropy).item()

def get_data(batch_size, dim, complexity="simple"):
    x = torch.randn(batch_size, dim)
    if complexity == "simple": y = (x[:, :50].sum(1, keepdim=True) > 0).float()
    else: y = (torch.sin(x[:, :200]).sum(1, keepdim=True) > 0).float()
    return x, y

def get_current_ranks(model):
    with torch.no_grad():
        return [round(get_eff_rank(model[i].linear.weight), 1) for i in range(12)]

save_path = r"C:\Users\Dima\fisn\conos"
os.makedirs(save_path, exist_ok=True)
dim = 1000

print("🏗️ Building ORTHOGONAL model: 12 layers, 1000x1000...")
layers = [OrthoMLPLayer(dim) for _ in range(12)]
model = nn.Sequential(*layers, nn.Linear(dim, 1), nn.Sigmoid())
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

try:
    # Код фаз полностью повторяет скрипт HE выше
    # PHASE 1: WARMUP
    print("\n--- PHASE 1: WARMUP (Layers 1-2) ---")
    for param in model.parameters(): param.requires_grad = False
    for i in range(2): 
        for param in model[i].parameters(): param.requires_grad = True
    for step in range(300):
        x, y = get_data(32, dim, "simple")
        optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        if step % 100 == 0: print(f"Step {step} | Loss: {loss.item():.4f} | Ranks: {get_current_ranks(model)}")

    # PHASE 2: SLIDING WINDOW
    print("\n--- PHASE 2: SLIDING WINDOW (4-layer Relay) ---")
    for start_idx in range(0, 10): 
        for param in model.parameters(): param.requires_grad = False
        for i in range(start_idx, min(start_idx + 4, 12)):
            for param in model[i].parameters(): param.requires_grad = True
        for param in model.parameters(): param.requires_grad = True
        for step in range(100):
            x, y = get_data(32, dim, "hard")
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        print(f"Window {start_idx+1} | Loss: {loss.item():.4f} | Ranks L1/L6/L12: {[get_current_ranks(model)[0], get_current_ranks(model)[5], get_current_ranks(model)[11]]}")
        gc.collect()

    # PHASE 3: FINAL POLISH
    print("\n--- PHASE 3: FINAL POLISH ---")
    for param in model.parameters(): param.requires_grad = True
    for step in range(750):
        x, y = get_data(32, dim, "hard")
        optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        if step % 100 == 0: print(f"Step {step} | Loss: {loss.item():.4f} | Ranks: {get_current_ranks(model)}")

    print("\n✅ ORTHOGONAL COMPLETE")
except Exception as e: print(f"\n❌ Error: {e}")

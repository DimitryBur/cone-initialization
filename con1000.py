import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import gc

# --- 1. ГЕОМЕТРИЧЕСКИЙ СЛОЙ ---
class ConeMLPLayer(nn.Module):
    def __init__(self, dim, angle_deg, high_val, low_val):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.init_cone_weights(angle_deg, high_val, low_val)

    def init_cone_weights(self, angle_deg, high_val, low_val):
        out_d, in_d = self.linear.weight.shape
        angle_rad = np.radians(angle_deg)
        with torch.no_grad():
            center_vec = torch.randn(in_d)
            center_vec /= center_vec.norm()
            num_high = int(out_d * 0.1) # 10% магистралей
            amplitudes = torch.full((out_d,), low_val)
            amplitudes[:num_high] = high_val
            scale = np.sqrt(2.0 / in_d)
            amplitudes *= scale 
            for i in range(out_d):
                noise = torch.randn(in_d)
                ortho_noise = noise - torch.dot(noise, center_vec) * center_vec
                ortho_noise /= (ortho_noise.norm() + 1e-8)
                w = torch.cos(torch.tensor(angle_rad)) * center_vec + \
                    torch.sin(torch.tensor(angle_rad)) * ortho_noise
                if i >= num_high and torch.rand(1) < 0.3: w *= -1
                self.linear.weight[i] = w * amplitudes[i]

    def forward(self, x):
        # Обычный проход (без skip-connection, чтобы проверить чистую эстафету)
        return self.relu(self.ln(self.linear(x)))

# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_eff_rank(tensor):
    with torch.no_grad():
        s = torch.linalg.svdvals(tensor)
        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        return torch.exp(entropy).item()

def get_data(batch_size, dim, complexity="simple"):
    x = torch.randn(batch_size, dim)
    if complexity == "simple":
        y = (x[:, :50].sum(1, keepdim=True) > 0).float()
    else:
        y = (torch.sin(x[:, :200]).sum(1, keepdim=True) > 0).float()
    return x, y

def get_current_ranks(model):
    ranks = []
    with torch.no_grad():
        # Проходим по всем 12 слоям MLP (исключая финальный классификатор)
        for i in range(12):
            r = get_eff_rank(model[i].linear.weight)
            ranks.append(round(r, 1))
    return ranks


# --- 3. НАСТРОЙКИ ---
save_path = r"C:\Users\Dima\fisn\conos"
os.makedirs(save_path, exist_ok=True)
dim = 1000
# Твои углы Линзы
angles = [15, 17, 25, 30, 40, 50, 60, 70, 73, 77, 81, 85]
history = []

print(f"🏗️ Building LENS: 12 layers, {dim}x{dim}...")
layers = []
for i in range(12):
    h = max(0.1, 0.7 - (i * 0.05))
    l = min(0.5, 0.1 + (i * 0.03))
    layers.append(ConeMLPLayer(dim, angles[i], h, l))

# Собираем модель + голову
model = nn.Sequential(*layers, nn.Linear(dim, 1), nn.Sigmoid())
optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Возвращаем 1e-3 для эстафеты
criterion = nn.BCELoss()

# --- 4. ПРОЦЕСС ОБУЧЕНИЯ (ЭСТАФЕТА) ---

try:
    # --- ШАГ 1: WARMUP (Только первые 2 слоя) ---
    print("\n--- PHASE 1: WARMUP (Layers 1-2) ---")
    for param in model.parameters(): param.requires_grad = False
    for i in range(2): 
        for param in model[i].parameters(): param.requires_grad = True
    
    for step in range(300):
        x, y = get_data(32, dim, "simple")
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Ranks: {get_current_ranks(model)}")

    # --- ШАГ 2: SLIDING WINDOW (3 слоя, 100 шагов на позицию) ---
    print("\n--- PHASE 2: SLIDING WINDOW (4-layer Relay) ---")
    for start_idx in range(0, 10): 
        print(f"-> Window: Layers {start_idx+1} to {start_idx+4} active")
        for param in model.parameters(): param.requires_grad = False
        # Включаем 3 слоя окна + финальную голову классификатора
        for i in range(start_idx, start_idx + 4):
            for param in model[i].parameters(): param.requires_grad = True
        for param in model[12].parameters(): param.requires_grad = True # Голова всегда активна
        
        for step in range(200):
            x, y = get_data(32, dim, "hard")
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        ranks = get_current_ranks(model)
        print(f"Window Done | Loss: {loss.item():.4f} | Ranks: {ranks}")
        history.append({"window": start_idx, "loss": loss.item(), "ranks": ranks})
        gc.collect()

    # --- ШАГ 3: FINAL FULL TRAINING (Все слои) ---
    print("\n--- PHASE 3: FINAL POLISH (All Layers) ---")
    for param in model.parameters(): param.requires_grad = True
    for step in range(800):
        x, y = get_data(32, dim, "hard")
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            all_ranks = get_current_ranks(model)
            print(f"Step {step:4} | Loss: {loss.item():.4f}")
            # Выводим ранги группами по 6 для удобства чтения
            print(f"   Ranks L1-L6:  {all_ranks[:6]}")
            print(f"   Ranks L7-L12: {all_ranks[6:]}")
            
            history.append({
                "step": step, 
                "loss": loss.item(), 
                "all_ranks": all_ranks
            })
            gc.collect()


    # СОХРАНЕНИЕ
    with open(os.path.join(save_path, "relay_12_stats.json"), "w") as f:
        json.dump(history, f)
    print(f"\n✅ Эстафета завершена успешно!")

except Exception as e:
    print(f"\n❌ Ошибка: {e}")

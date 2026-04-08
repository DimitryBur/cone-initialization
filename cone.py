import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json


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
            num_high = int(out_d * 0.1)
            amplitudes = torch.full((out_d,), low_val)
            amplitudes[:num_high] = high_val
            scale = 1.0 / np.sqrt(in_d)
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
        return self.relu(self.ln(self.linear(x)))

def get_eff_rank(tensor):
    with torch.no_grad():
        s = torch.linalg.svdvals(tensor)
        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        return torch.exp(entropy).item()


save_path = r"C:\Users\Dima\fisn\conos"
os.makedirs(save_path, exist_ok=True)
dim = 200
angles = [5, 12, 25, 30, 35, 40] 
history = []

# Сборка модели: 6 слоев + голова
layers = []
for i in range(6):
    h, l = 0.7 - (i*0.1), 0.1 + (i*0.04)
    layers.append(ConeMLPLayer(dim, angles[i], h, l))


model = nn.Sequential(*layers, nn.Linear(dim, 1), nn.Sigmoid())

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

def run_phase(name, steps, complexity):
    print(f"\n--- Phase: {name} ---")
    for step in range(steps):
        x = torch.randn(64, dim)
        # Гарантируем, что y имеет форму (64, 1)
        if complexity == "simple": 
            y = (x[:, :5].sum(1, keepdim=True) > 0).float()
        else: 
            y = (torch.sin(x[:, :10]).sum(1, keepdim=True) > 0).float()

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y) 
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                ranks = [get_eff_rank(model[i].linear.weight) for i in range(6)]
            log_entry = {
                "phase": name, "step": step, "loss": round(loss.item(), 4), "ranks": [round(r, 2) for r in ranks]
            }
            history.append(log_entry)
            print(f"Step {step} | Loss: {log_entry['loss']} | Ranks: {log_entry['ranks']}")


try:
    run_phase("Warmup", 500, "simple")
    run_phase("Training", 1000, "hard")
    
    with open(os.path.join(save_path, "train_stats.json"), "w") as f:
        json.dump(history, f)
    print(f"\n✅ Успех! Файл в {save_path}")
except Exception as e:
    print(f"\n❌ Ошибка: {e}")

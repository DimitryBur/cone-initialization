import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json


class XavierMLPLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.relu(self.ln(self.linear(x)))


def get_eff_rank(tensor):
    with torch.no_grad():
        в
        s = torch.linalg.svdvals(tensor)
        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        return torch.exp(entropy).item()


save_path = r"C:\Users\Dima\fisn\conos"
os.makedirs(save_path, exist_ok=True)
dim = 200
history = []


layers = []
for _ in range(6):
    layers.append(XavierMLPLayer(dim))

model = nn.Sequential(
    *layers, 
    nn.Linear(dim, 1), 
    nn.Sigmoid()
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()


def run_phase(name, steps, complexity):
    print(f"\n--- Phase: {name} (XAVIER) ---")
    for step in range(steps):
        # Генерация данных (идентично предыдущему тесту)
        x = torch.randn(64, dim)
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
                "phase": name, 
                "step": step, 
                "loss": round(loss.item(), 4), 
                "ranks": [round(r, 2) for r in ranks]
            }
            history.append(log_entry)
            print(f"Step {step} | Loss: {log_entry['loss']} | Ranks: {log_entry['ranks']}")


try:
    run_phase("Warmup", 500, "simple")
    run_phase("Training", 1000, "hard")
    
    
    stats_file = os.path.join(save_path, "xavier_stats.json")
    with open(stats_file, "w") as f:
        json.dump(history, f)
        
    
    torch.save(model.state_dict(), os.path.join(save_path, "xavier_model.pt"))
    
    print(f"\n✅ Xavier тест завершен!")
    print(f"Результаты записаны в: {stats_file}")

except Exception as e:
    print(f"\n❌ Ошибка при выполнении: {e}")


print("\n💡 Сравни эти цифры с логами Cone-Hierarchy:")
print("1. Насколько высоки ранги в начале (Step 0 Warmup)?")
print("2. Каков финальный Loss на шаге 950 Training?")
print("3. Был ли скачок Loss при переходе к фазе Training?")

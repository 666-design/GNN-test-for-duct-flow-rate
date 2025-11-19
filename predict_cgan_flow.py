import torch, torch.nn as nn
import numpy as np, pandas as pd, joblib

# 载入归一化器与生成器
x_scaler = joblib.load("cgan_x_scaler.pkl")
y_scaler = joblib.load("cgan_y_scaler.pkl")

class Generator(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+z_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, y_dim)
        )
    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=1))

x_dim, z_dim, y_dim = 2, 8, 2
device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator(x_dim, z_dim, y_dim).to(device)
G.load_state_dict(torch.load("cgan_G.pth", map_location=device))
G.eval()

# 准备输入（可以多行一起输）
X_new = pd.DataFrame({
    "inlet_velocity": [10.0, 9.2],
    "degree"        : [30.0, 45.0]
})

X_norm = x_scaler.transform(X_new.astype(np.float32))
X_t    = torch.tensor(X_norm, dtype=torch.float32).to(device)

# 生成预测（多样性：每行可采多次 z）───────────
n_sample = 3                     # 每个输入生成 3 个候选
pred_list = []
for _ in range(n_sample):
    z = torch.randn(len(X_new), z_dim).to(device)
    y_norm = G(X_t, z).cpu().detach().numpy()
    y      = y_scaler.inverse_transform(y_norm)
    pred_list.append(y)

# 取均值作为最终预测，也可输出区间
y_mean = np.mean(pred_list, axis=0)
result = pd.DataFrame(
    np.hstack([X_new, y_mean]),
    columns=["inlet_velocity","degree",
             "main_mass_flow_pred","branch_mass_flow_pred"]
)
print(result.to_string(index=False, float_format="%.3f"))
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt, json, random
import joblib

SEED = 168
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

#1. 读数据
df = pd.read_csv("pipe_data.csv")
X = df[["inlet_velocity", "degree"]].values.astype(np.float32)
y = df[["main_mass_flow", "branch_mass_flow"]].values.astype(np.float32)

#2. 标准化
x_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)
Xn = x_scaler.transform(X)
yn = y_scaler.transform(y)

#3. 数据集（80/20）
idx = np.arange(len(X)); np.random.shuffle(idx)
split = int(0.8*len(X))
train_idx, test_idx = idx[:split], idx[split:]
X_train, y_train = Xn[train_idx], yn[train_idx]
X_test,  y_test  = Xn[test_idx], yn[test_idx]

device = "cuda" if torch.cuda.is_available() else "cpu"

#4. 网络
class Generator(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+z_dim, 128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64, y_dim)
        )
    def forward(self, x, z):
        inp = torch.cat([x,z], dim=1)
        return self.net(inp)

class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim+y_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128,64), nn.LeakyReLU(0.2),
            nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self, x, y):
        inp = torch.cat([x,y], dim=1)
        return self.net(inp)

x_dim, z_dim, y_dim = 2, 8, 2
G = Generator(x_dim,z_dim,y_dim).to(device)
D = Discriminator(x_dim,y_dim).to(device)

opt_G = optim.Adam(G.parameters(), lr=2e-4)
opt_D = optim.Adam(D.parameters(), lr=2e-4)
bce = nn.BCELoss()

batch = 128
def get_batch():
    idx = np.random.randint(0, len(X_train), batch)
    return torch.tensor(X_train[idx]).to(device), torch.tensor(y_train[idx]).to(device)

#5. 训练
steps = 5000
for step in range(steps):
    # ----- train D -----
    Xb, yb = get_batch()
    z = torch.randn(batch, z_dim).to(device)
    y_fake = G(Xb, z).detach()
    D_real = D(Xb, yb)
    D_fake = D(Xb, y_fake)
    loss_D = bce(D_real, torch.ones_like(D_real)) + \
             bce(D_fake, torch.zeros_like(D_fake))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # ----- train G -----
    z = torch.randn(batch, z_dim).to(device)
    y_fake = G(Xb, z)
    D_fake = D(Xb, y_fake)
    loss_G = bce(D_fake, torch.ones_like(D_fake))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    if (step+1)%1000==0:
        print(f"step {step+1}/{steps}  loss_D {loss_D.item():.3f}  loss_G {loss_G.item():.3f}")

#6. 评估
X_test_t = torch.tensor(X_test).to(device)
z = torch.randn(len(X_test), z_dim).to(device)
y_pred_n = G(X_test_t, z).cpu().detach().numpy()
y_pred   = y_scaler.inverse_transform(y_pred_n)

mre  = mean_absolute_percentage_error(df.iloc[test_idx][["main_mass_flow","branch_mass_flow"]], y_pred)
r2   = r2_score(df.iloc[test_idx][["main_mass_flow","branch_mass_flow"]], y_pred)
print(f"\nMRE={mre:.4f}  R2={r2:.4f}")

#7. 保存结果
torch.save(G.state_dict(),"cgan_G.pth")
with open("cgan_metrics.json","w") as f:
    json.dump({"mre":float(mre),"r2":float(r2)},f,indent=2)

#8. 散点
plt.scatter(df.iloc[test_idx]["main_mass_flow"], y_pred[:,0], s=10)
plt.xlabel("True Main"); plt.ylabel("Pred Main"); plt.title("cGAN Main Flow")
plt.tight_layout(); plt.savefig("cgan_scatter.png")

joblib.dump(x_scaler, "cgan_x_scaler.pkl")
joblib.dump(y_scaler, "cgan_y_scaler.pkl")

print("GAN训练结束，模型与评估已生成")


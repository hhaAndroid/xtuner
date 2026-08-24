import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 超参数
dim = 2   # 数据维度（2D点）
num_samples = 1000
num_steps = 50  # ODE求解步数
lr = 1e-3
epochs = 5000

# 目标分布：正弦曲线上的点（x1坐标）
x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # 0到4π (1000,1)
y1_samples = torch.sin(x1_samples)                      # y=sin(x)
target_data = torch.cat([x1_samples, y1_samples], dim=1) # shape (1000,2)

# 噪声分布：高斯噪声（x0坐标）
noise_data = torch.randn(num_samples, dim) * 2  # shape (1000,2)

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入维度: x (2) + t (1) = 3
            nn.ReLU(),
            nn.Linear(64, dim)
        )
  
    def forward(self, x, t):
        # 直接拼接x和t（t的形状需为(batch_size, 1)）
        return self.net(torch.cat([x, t], dim=1))
        
model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # 随机采样噪声点和目标点
    idx = torch.randperm(num_samples)
    x0 = noise_data[idx]  # 起点：噪声 shape (1000,2)
    x1 = target_data[idx] # 终点：正弦曲线

    # 时间t的形状为 (batch_size, 1)
    t = torch.rand(x0.size(0), 1)  # 例如：shape (1000, 1)
  
    # 线性插值生成中间点
    xt = (1 - t) * x0 + t * x1
  
    # 模型预测向量场（直接传入t，无需squeeze）
    vt_pred = model(xt, t)  # t的维度保持不变 shape (1000,2)
  
    # 目标向量场：x1 - x0
    vt_target = x1 - x0 # shape (1000,2)
  
    # 损失函数
    loss = torch.mean((vt_pred - vt_target)**2)
  
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ============ 绘制向量场 ============
# 创建网格点
x_min, x_max = -6, 6
y_min, y_max = -3, 3
grid_density = 20  # 网格密度

x_grid = np.linspace(x_min, x_max, grid_density)
y_grid = np.linspace(y_min, y_max, grid_density)
X, Y = np.meshgrid(x_grid, y_grid)

# 在t=0时刻计算向量场（从噪声开始）
grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
t_grid = torch.zeros(grid_points.shape[0], 1)  # t=0

with torch.no_grad():
    vectors = model(grid_points, t_grid).numpy()

U = vectors[:, 0].reshape(X.shape)  # x方向分量
V = vectors[:, 1].reshape(X.shape)  # y方向分量

# ============ 生成单条轨迹 ============
x = noise_data[0:1]  # 选择初始噪声点进行轨迹生成
trajectory = [x.detach().numpy()]

t = 0
delta_t = 1 / num_steps
with torch.no_grad():
    for i in range(num_steps):
        vt = model(x, torch.tensor([[t]], dtype=torch.float32))
        t += delta_t
        x = x + vt * delta_t
        trajectory.append(x.detach().numpy())

trajectory = np.array(trajectory).squeeze()

# ============ 绘制所有1000个噪声的轨迹 ============
all_trajectories = []
for i in range(num_samples):
    x = noise_data[i:i+1]
    traj = [x.detach().numpy()]
    
    t = 0
    with torch.no_grad():
        for step in range(num_steps):
            vt = model(x, torch.tensor([[t]], dtype=torch.float32))
            t += delta_t
            x = x + vt * delta_t
            traj.append(x.detach().numpy())
    
    all_trajectories.append(np.array(traj).squeeze())

# ============ 绘图 ============
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 左图：向量场 + 单条轨迹
ax1 = axes[0]
ax1.quiver(X, Y, U, V, alpha=0.6, scale=20, width=0.003, color='gray')
ax1.scatter(target_data[:,0], target_data[:,1], c='blue', s=10, label='Target (sin(x))', alpha=0.5)
ax1.scatter(noise_data[:,0], noise_data[:,1], c='red', s=10, alpha=0.3, label='Noise')
ax1.plot(trajectory[:,0], trajectory[:,1], 'g-', linewidth=2, label='Single Generated Path')
# 绘制训练数据在 t=0 时候的向量场，当然也可以绘制在 t=1 时候的向量场
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.legend()
ax1.set_title("Vector Field at t=0 with Single Trajectory")
ax1.grid(True, alpha=0.3)

# 右图：所有1000条轨迹
ax2 = axes[1]
ax2.scatter(target_data[:,0], target_data[:,1], c='blue', s=10, label='Target (sin(x))', alpha=0.5)
ax2.scatter(noise_data[:,0], noise_data[:,1], c='red', s=10, alpha=0.3, label='Noise Start')

# 绘制所有轨迹
for traj in all_trajectories:
    ax2.plot(traj[:,0], traj[:,1], 'green', alpha=0.1, linewidth=0.5)

# 绘制终点
final_points = np.array([traj[-1] for traj in all_trajectories])
ax2.scatter(final_points[:,0], final_points[:,1], c='orange', s=10, alpha=0.5, label='Generated End Points')

ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.legend()
ax2.set_title("All 1000 Trajectories from Noise to Target")
ax2.grid(True, alpha=0.3)

# 绘制 1000 个训练点在推理时候的向量场
plt.tight_layout()
plt.savefig("flow_matching_vector_field.png", dpi=150)

print(f"Final point of first trajectory: {trajectory[-1]}")
print(f"Normalized: {trajectory[-1] / (torch.pi / 10 * 4)}")
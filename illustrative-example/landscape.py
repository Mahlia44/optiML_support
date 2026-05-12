import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

x_boundary = 4.5
y_boundary = 4.5

z_lim = 100

x_low_lim = -x_boundary/1.8
y_low_lim = -y_boundary/1.8

# --- LANDSCAPE ----

def landscape(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = (x*y - 1) ** 2
    return np.minimum(z, z_lim)

# Create surface data
xs = np.linspace(x_low_lim, x_boundary, 400)
ys = np.linspace(y_low_lim, y_boundary, 400)
xs, ys = np.meshgrid(xs, ys)
zs = landscape(xs, ys)

# Define the manifold of minima
x_minima = np.linspace(1 / x_boundary, x_boundary, 400)
y_minima = 1 / x_minima
z_minima = landscape(x_minima, y_minima)

# Minimal norm solution
x_min_norm = 1
y_min_norm = 1
z_min_norm = landscape(x_min_norm, y_min_norm)

# --- INITIALIZATION ---

x_init = 4.0
y_init = 2.0
z_init = landscape(x_init, y_init)

# --- SYNTH DATASET ---

n = 100
x = np.random.randn(n)
x = x / np.sqrt(np.mean(x**2)) # sum_i x_i^2 = n
y = (1 / np.mean(x)) * np.ones(n) # sum_i x_i y_i = n

xt = torch.tensor(x, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(xt, yt)

bs = int(n/10)
dataloader_gd = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
dataloader_sgd = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
dataloader_gdwd = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
dataloader_adam = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

# --- TORCH MODEL ----

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(x_init))
        self.b = torch.nn.Parameter(torch.tensor(y_init))
    
    def forward(self, x):
        return self.a * self.b * x

mod_gd = LinearModel()
mod_sgd = LinearModel()
mod_gdwd = LinearModel()
mod_adam = LinearModel()

# --- OPTIMIZER ---

lr = 0.008
wd = 1.0

opt_gd = torch.optim.SGD(mod_gd.parameters(), lr=lr)
opt_sgd = torch.optim.SGD(mod_sgd.parameters(), lr=lr)
opt_gdwd = torch.optim.SGD(mod_gdwd.parameters(), lr=lr, weight_decay=wd)
opt_adam = torch.optim.Adam(mod_adam.parameters(), lr=lr*10)
crit = torch.nn.MSELoss()

# --- TRAJECTORIES ---

n_iters = 550

def trajectory(model, dataloader, opt, crit, n_iters):
    iters = 0
    traj = [(model.a.item(), model.b.item())] # optimizer trajectory
    while iters < n_iters:
        for (x,y) in dataloader:
            if iters == n_iters: break
            opt.zero_grad()
            yhat = model(x)
            loss = crit(yhat, y)
            loss.backward()
            opt.step()
            traj.append((model.a.item(), model.b.item()))
            iters += 1
    return traj

traj_gd = trajectory(mod_gd, dataloader_gd, opt_gd, crit, n_iters)
traj_sgd = trajectory(mod_sgd, dataloader_sgd, opt_sgd, crit, n_iters)
traj_gdwd = trajectory(mod_gdwd, dataloader_gdwd, opt_gdwd, crit, n_iters)
traj_adam = trajectory(mod_adam, dataloader_adam, opt_adam, crit, n_iters)

# unzip x,y trajectories!
x_traj_gd, y_traj_gd = zip(*traj_gd)
x_traj_sgd, y_traj_sgd = zip(*traj_sgd)
x_traj_gdwd, y_traj_gdwd = zip(*traj_gdwd)
x_traj_adam, y_traj_adam = zip(*traj_adam)

x_traj_gd, y_traj_gd = np.array(x_traj_gd), np.array(y_traj_gd)
x_traj_sgd, y_traj_sgd = np.array(x_traj_sgd), np.array(y_traj_sgd)
x_traj_gdwd, y_traj_gdwd = np.array(x_traj_gdwd), np.array(y_traj_gdwd)
x_traj_adam, y_traj_adam = np.array(x_traj_adam), np.array(y_traj_adam)

z_traj_gd = landscape(x_traj_gd, y_traj_gd)
z_traj_sgd = landscape(x_traj_sgd, y_traj_sgd)
z_traj_gdwd = landscape(x_traj_gdwd, y_traj_gdwd)
z_traj_adam = landscape(x_traj_adam, y_traj_adam)

# --- PLOT ---

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, zs, cmap='magma', zorder=0)
ax.plot(x_minima, y_minima, z_minima, zorder=10, color='white')
ax.plot(-x_minima, -y_minima, z_minima, zorder=10, color='white')
ax.plot(0, 0, landscape(0,0), marker='o', markersize=3, color='white', zorder=10)
ax.plot(x_min_norm, y_min_norm, z_min_norm, marker='o', markersize=3, color='magenta', zorder=10)
ax.plot(-x_min_norm, -y_min_norm, z_min_norm, marker='o', markersize=3, color='magenta', zorder=10)

ax.plot(x_init, y_init, z_init, marker='o', markersize=3, color='white', zorder=10)
ax.plot(x_traj_gd, y_traj_gd, z_traj_gd, linestyle='-', marker='o', markersize=1, linewidth=0.8, color='red', zorder=10, label='gd')
ax.plot(x_traj_gdwd, y_traj_gdwd, z_traj_gdwd, linestyle='-', marker='o', markersize=0.5, linewidth=0.8, color='gold', zorder=10, label='gdwd')
ax.plot(x_traj_sgd, y_traj_sgd, z_traj_sgd, linestyle='-', linewidth=0.8, color='forestgreen', zorder=10, label='sgd')
ax.plot(x_traj_adam, y_traj_adam, z_traj_adam, linestyle='-', linewidth=0.8, color='blue', zorder=10, label='adam')

# Labels and title
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('loss')

ax.set_xlim(x_low_lim, x_boundary)
ax.set_ylim(y_low_lim, y_boundary)
ax.set_zlim(0, z_lim)

ax.view_init(elev=30, azim=-125)  # elev is the elevation angle
ax.dist = 8  # Smaller values bring the plot "closer" (zoom in)

#ax.view_init(elev=50, azim=-140)  # elev is the elevation angle
#ax.dist = 5  # Smaller values bring the plot "closer" (zoom in)

# Display the plot
plt.tight_layout()
plt.show()


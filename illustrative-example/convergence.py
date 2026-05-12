import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# --- SYNTH DATASET ---

# D1 oscillates on inputs
D1_x = torch.tensor([[1.0], [1.0]])
D1_y = torch.tensor([-1.0, 1.0])

# D2 oscillates on inputs
D2_x = torch.tensor([[1/3], [3.0]])
D2_y = torch.tensor([0.0, 0.0])

ds01 = torch.utils.data.TensorDataset(D1_x, D1_y)
ds02 = torch.utils.data.TensorDataset(D2_x, D2_y)

dataloader_gd = torch.utils.data.DataLoader(ds01, batch_size=2, shuffle=False)
dataloader_sgd01 = torch.utils.data.DataLoader(ds01, batch_size=1, shuffle=False)
dataloader_sgd02 = torch.utils.data.DataLoader(ds02, batch_size=1, shuffle=False)

# --- TORCH MODEL ----

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(4.0))
        self.b = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return self.a * self.b * x

mod_gd = LinearModel()
mod_sgd01 = LinearModel()
mod_sgd02 = LinearModel()

# --- OPTIMIZER ---

lr = 0.008
opt_gd = torch.optim.SGD(mod_gd.parameters(), lr=lr)
opt_sgd01 = torch.optim.SGD(mod_sgd01.parameters(), lr=lr)
opt_sgd02 = torch.optim.SGD(mod_sgd02.parameters(), lr=lr)
crit = torch.nn.MSELoss()

# --- TRAJECTORIES ---

n_iters = 500

def trajectory(model, dataloader, opt, crit, n_iters):
    iters = 0
    losses = []
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
            losses.append(loss.item())
            iters += 1
    return traj, losses

traj_gd, loss_gd = trajectory(mod_gd, dataloader_gd, opt_gd, crit, n_iters)
traj_sgd01, loss_sgd01 = trajectory(mod_sgd01, dataloader_sgd01, opt_sgd01, crit, n_iters)
traj_sgd02, loss_sgd02 = trajectory(mod_sgd02, dataloader_sgd02, opt_sgd02, crit, n_iters)

# unzip x,y trajectories!
x_traj_gd, y_traj_gd = zip(*traj_gd)
x_traj_sgd, y_traj_sgd = zip(*traj_sgd01)
x_traj_gdwd, y_traj_gdwd = zip(*traj_sgd02)

x_traj_gd, y_traj_gd = np.array(x_traj_gd), np.array(y_traj_gd)
x_traj_sgd, y_traj_sgd = np.array(x_traj_sgd), np.array(y_traj_sgd)
x_traj_gdwd, y_traj_gdwd = np.array(x_traj_gdwd), np.array(y_traj_gdwd)

# --- PLOT ---

fig = plt.figure()
plt.plot(loss_gd, label='gd')
plt.plot(loss_sgd01, label='sgd d1')
plt.plot(loss_sgd02, label='sgd d2')
plt.yscale('log')
plt.legend()
plt.show()
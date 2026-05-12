from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.opt = optimizer
        self.crit = criterion

    def callbacks(self, metrics, prev_params=None):
        for i, (name, param) in enumerate(self.model.named_parameters()):
            metrics[name + ".params"].append(param.tolist())
            metrics[name + ".norm"].append(param.norm().item())
            
            if prev_params is not None:
                metrics[name + ".delta"].append((param - prev_params[i]).norm().item())

            # this could be averaged over the batch
            if param.grad is not None:
                metrics[name + ".grad"].append(param.grad.tolist())
                metrics[name + ".grad.norm"].append(param.grad.norm().item())

            if "weight" in name and param.ndim == 2:
                with torch.no_grad():
                    weights = param.clone().detach()
                    eps, tol = 0, 1e-3

                    # normalize weight matrix!
                    #weights[weights.abs() < eps] = 0
                    #weights = weights / torch.norm(weights)

                    # normalized / stable rank
                    rank = torch.linalg.matrix_rank(weights, tol=tol)
                    metrics[name + ".rank"].append(rank.item())

                    # Gram matrix
                    gram = (weights.T @ weights).numpy()
                    metrics[name + ".gram"].append(gram.tolist())

    def callgram(self, metrics):
        for i, (name, param) in enumerate(self.model.named_parameters()):

            if "weight" in name and param.ndim == 2 and '0' in name:
                with torch.no_grad():
                    weights = param.clone().detach()
                    eps, tol = 0, 1e-3
                    # normalize weight matrix!
                    #weights[weights.abs() < eps] = 0
                    #weights = weights / torch.norm(weights)
                    # Gram matrix
                    gram = (weights.T @ weights).numpy()
                    # Diag of the Gram matrix
                    diag = torch.diag(torch.from_numpy(gram)).numpy()
                    metrics[name + ".gram.diag"].append(diag.tolist())

    def train(
        self,
        n_epochs: int,
        max_iters: Optional[int | None] = None,
        freq_callbacks: int = 1,
        clip_value: float = -1.0,
    ):
        # initialize counters
        iters_per_epoch = len(self.dataloader)  # num batches per epoch
        n_iters_tot = iters_per_epoch * n_epochs
        iters_idx = 0
        
        self.model.train()
        pbar = tqdm(range(n_iters_tot))
        metrics = defaultdict(list)
        self.callbacks(metrics) # init snapshot.
        self.callgram(metrics) # init snapshot.

        for epoch_idx in range(n_epochs):
            epoch_loss = 0.
            
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                if max_iters is not None and iters_idx >= max_iters:
                    break
                
                # forward pass
                outputs = self.model(inputs)

                # backward pass
                self.opt.zero_grad()
                loss = self.crit(outputs, targets)
                loss.backward()
                if clip_value > 0:
                    clip_grad_norm_(self.model.parameters(), clip_value)
                self.opt.step()

                # log iter-level metrics
                epoch_loss += loss.item()
                metrics["train_loss_per_iter"].append(loss.item())

                #if freq_callbacks > 0 and batch_idx % freq_callbacks == -1:
                    #self.callbacks(metrics)

                if freq_callbacks > 0 and batch_idx % freq_callbacks == 0:
                    self.callgram(metrics)

                # move progress bar
                pbar.set_description(f"epoch {epoch_idx+1}/{n_epochs} iter {iters_idx+1}/{n_iters_tot} | train loss {loss.item():.3f}")
                pbar.update(1)
                iters_idx += 1

            # log epoch-level metrics
            metrics["train_loss_per_epoch"].append(epoch_loss / iters_per_epoch)

        pbar.close()

        self.callbacks(metrics) # final snapshot.
        metrics["n_epochs"] = epoch_idx
        metrics["n_iters"] = iters_idx

        return dict(metrics)

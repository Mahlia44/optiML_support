#!/usr/bin/env python3
"""
Train an MLP on a synthetic dataset with configurable optimizer and target function.
Saves model weights and training curves for later visualization.

Usage examples
--------------
# Full-batch gradient descent, linear target:
python train_baseline.py --optimizer gd --target linear --n_iters 20000

# Mini-batch SGD, multiple batch sizes:
python train_baseline.py --optimizer sgd --batch_size 512 --n_iters 200000 --n_trajs 5
python train_baseline.py --optimizer sgd --batch_size 128 --n_iters 200000 --n_trajs 5
python train_baseline.py --optimizer sgd --batch_size 32  --n_iters 200000 --n_trajs 5
python train_baseline.py --optimizer sgd --batch_size 1   --n_iters 200000 --n_trajs 5

# GD with weight decay (gold baseline):
python train_baseline.py --optimizer gd --weight_decay 0.1 --n_iters 5000

# Adam / Muon:
python train_baseline.py --optimizer adam --lr 1e-3 --target sine
python train_baseline.py --optimizer muon --lr 0.02 --target staircase

Outputs (in --output_dir)
--------------------------
{run_name}.npz         – numpy arrays: weights (init/post) + loss/irelnorm curves
{run_name}_config.json – full hyperparameter config
"""

import argparse
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from source.models import MLP


# ── Reproducibility ───────────────────────────────────────────────────────────

def seedall(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# ── Synthetic data ────────────────────────────────────────────────────────────

def make_data(
    target: str,
    n: int,
    d: int,
    r: int,
    seed: int,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (xt, yt) for the chosen target function.

    Data follows the sparse-support setup: only the first r of d features are
    relevant; irrelevant features are centred. The dataset is doubled via a
    ±eps trick (y1, y2) to break trivial symmetry.
    """
    seedall(seed)
    x = np.random.randn(n, d)
    x[:, r:] -= x[:, r:].mean(axis=0)   # centre irrelevant features

    W = np.zeros(d)
    W[:r] = 1.0
    eps = noise * np.random.randn(n)
    xW = x @ W

    if target == "linear":
        raw = xW
    elif target == "sine":
        raw = np.sin(xW)
    elif target == "staircase":
        raw = sum(W[i] * np.power(x[:, i], i + 1) for i in range(r))
    else:
        raise ValueError(f"Unknown target '{target}'. Choose: linear | sine | staircase")

    y1 = (raw + eps).reshape(-1, 1)
    y2 = (raw - eps).reshape(-1, 1)
    x = np.concatenate([x, x], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ── Muon optimizer ────────────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """Momentum Orthogonal Update optimizer.

    For each parameter matrix G, the gradient momentum buffer is
    orthogonalized via Newton-Schulz iterations before the weight update.
    Scalar/vector parameters fall back to plain SGD with momentum.

    Reference: https://arxiv.org/abs/2409.20325
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(G: torch.Tensor, steps: int) -> torch.Tensor:
        """Newton-Schulz iteration: maps G → G / ||G||_op (approximately)."""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.float()
        X = X / (X.norm() + 1e-7)
        transposed = X.size(0) > X.size(1)
        if transposed:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X.to(G.dtype)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if nesterov else buf.clone()
                if update.ndim >= 2:
                    update = self._orthogonalize(update, ns_steps)
                    update = update * (update.numel() ** 0.5)
                p.data.add_(update, alpha=-lr)
        return loss


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(args, params) -> torch.optim.Optimizer:
    if args.optimizer in ("gd", "sgd"):
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "muon":
        return Muon(params, lr=args.lr, momentum=args.muon_momentum, ns_steps=args.muon_ns_steps)
    else:
        raise ValueError(f"Unknown optimizer '{args.optimizer}'")


# ── Weight helpers ────────────────────────────────────────────────────────────

def extract_weights(model: MLP) -> list[np.ndarray]:
    """Return a list of weight matrices (one per layer, detached)."""
    return [
        layer.linear_act_block.linear.weight.detach().clone().numpy()
        for layer in model.layers
    ]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_trajectory(
    model: MLP,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_iters: int,
    r: int,
    d: int,
    clip_value: float,
    seed_traj: int,
    traj_idx: int = 0,
    use_wandb: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    """Run one training trajectory; return (weights_init, weights_post, losses, irelnorms)."""
    seedall(seed_traj)
    weights_init = extract_weights(model)

    model.train()
    crit = nn.MSELoss()
    losses: list[float] = []
    irelnorms: list[float] = []

    pbar = tqdm(range(n_iters))
    iters_idx, epoch_idx = 0, 0

    while iters_idx < n_iters:
        for inputs, targets in dataloader:
            if iters_idx >= n_iters:
                break

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = crit(outputs, targets)
            loss.backward()
            if clip_value > 0:
                clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            losses.append(loss.item())

            W1 = model.layers[0].linear_act_block.linear.weight.detach().numpy()
            irelnorms.append(float(np.linalg.norm(W1[:, r:d])))

            if use_wandb:
                wandb.log({
                    f"traj{traj_idx}/loss": loss.item(),
                    f"traj{traj_idx}/irelnorm": irelnorms[-1],
                }, step=iters_idx)

            total_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            pbar.set_description(
                f"epoch {epoch_idx + 1} iter {iters_idx + 1}/{n_iters}"
                f" | loss {loss.item():.4f} | norm {total_norm:.3f}"
            )
            pbar.update(1)
            iters_idx += 1

        epoch_idx += 1

    pbar.close()
    weights_post = extract_weights(model)
    return weights_init, weights_post, losses, irelnorms


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_results(
    run_name: str,
    output_dir: str,
    config: dict,
    results: list[dict],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for traj_idx, res in enumerate(results):
        for layer_idx, w in enumerate(res["weights_init"]):
            arrays[f"traj{traj_idx}_layer{layer_idx}_init"] = w
        for layer_idx, w in enumerate(res["weights_post"]):
            arrays[f"traj{traj_idx}_layer{layer_idx}_post"] = w
        arrays[f"traj{traj_idx}_losses"] = np.array(res["losses"])
        arrays[f"traj{traj_idx}_irelnorms"] = np.array(res["irelnorms"])

    npz_path = os.path.join(output_dir, f"{run_name}.npz")
    np.savez(npz_path, **arrays)

    config_path = os.path.join(output_dir, f"{run_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  weights → {npz_path}")
    print(f"  config  → {config_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── core ──
    p.add_argument(
        "--optimizer", choices=["gd", "sgd", "adam", "adamw", "muon"], default="gd",
        help="Optimizer. 'gd' = full-batch gradient descent (ignores --batch_size).",
    )
    p.add_argument(
        "--target", choices=["linear", "sine", "staircase"], default="linear",
        help="Synthetic target function.",
    )
    p.add_argument("--n_iters", type=int, default=20_000, help="Total gradient steps.")
    p.add_argument("--n_trajs", type=int, default=1, help="Independent training trajectories.")

    # ── optimizer hparams ──
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay (SGD / Adam).")
    p.add_argument(
        "--batch_size", type=int, default=512,
        help="Mini-batch size (ignored when --optimizer gd).",
    )
    p.add_argument("--clip_value", type=float, default=1.0, help="Gradient clip norm (0 = disabled).")
    p.add_argument("--muon_momentum", type=float, default=0.95, help="Muon: momentum coefficient.")
    p.add_argument("--muon_ns_steps", type=int, default=5, help="Muon: Newton-Schulz iteration count.")

    # ── model ──
    p.add_argument("--hiddens", type=int, nargs="+", default=[15, 15, 15], help="Hidden layer widths.")
    p.add_argument("--relu", action="store_true", help="Use ReLU activations (default: linear MLP).")
    p.add_argument(
        "--init_method", default="he_normal",
        choices=["he_normal", "he_uniform", "xavier_normal", "xavier_uniform"],
        help="Weight initialisation scheme.",
    )

    # ── data ──
    p.add_argument("--n_samples", type=int, default=5000, help="Base dataset size (doubled internally).")
    p.add_argument("--input_dim", type=int, default=15, help="Input feature dimension d.")
    p.add_argument("--n_relevant", type=int, default=5, help="Number of relevant features r.")
    p.add_argument("--noise", type=float, default=0.001, help="Label noise std.")
    p.add_argument("--seed_init", type=int, default=0, help="Seed for data generation and model init.")

    # ── output ──
    p.add_argument("--output_dir", default="results", help="Directory to write results.")
    p.add_argument("--run_name", default=None, help="Override auto-generated run name.")

    # ── wandb ──
    p.add_argument("--wandb_entity", default=None,
                   help="Weights & Biases entity (username or team). Prompted interactively if omitted.")
    p.add_argument("--wandb_project", default="sgd-finds-support",
                   help="W&B project name (default: sgd-finds-support).")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")

    return p.parse_args()


def _auto_run_name(args: argparse.Namespace) -> str:
    opt = args.optimizer if args.optimizer == "gd" else f"{args.optimizer}{args.batch_size}"
    wd = f"_wd{args.weight_decay}" if args.weight_decay != 0.0 else ""
    return f"{args.target}_{opt}_lr{args.lr}{wd}_{args.n_iters}iters"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_name = args.run_name or _auto_run_name(args)
    d, r = args.input_dim, args.n_relevant

    print(f"\n=== {run_name} ===")
    print(f"  optimizer={args.optimizer}  target={args.target}  n_iters={args.n_iters}")
    print(f"  lr={args.lr}  wd={args.weight_decay}  bs={'full' if args.optimizer == 'gd' else args.batch_size}")
    print(f"  hiddens={args.hiddens}  relu={args.relu}  n_trajs={args.n_trajs}\n")

    # ── wandb init ──
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("wandb not installed — skipping logging. Run `uv add wandb` to enable it.")
    if use_wandb:
        entity = args.wandb_entity
        if entity is None:
            entity = input("wandb entity (your username or team name): ").strip()
        wandb.init(
            entity=entity,
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        print(f"  wandb run: {wandb.run.url}\n")

    xt, yt = make_data(args.target, args.n_samples, d, r, args.seed_init, args.noise)
    bs = len(xt) if args.optimizer == "gd" else args.batch_size
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt),
        batch_size=bs,
        shuffle=True,
    )

    results: list[dict] = []
    for traj in range(args.n_trajs):
        print(f"--- trajectory {traj + 1}/{args.n_trajs} ---")
        seedall(args.seed_init)
        model = MLP(
            in_features=d,
            hidden_features=args.hiddens,
            out_features=1,
            act_func=nn.ReLU() if args.relu else None,
            bias=False,
            init_method=args.init_method,
        )
        opt = build_optimizer(args, model.parameters())
        w_init, w_post, losses, irelnorms = train_trajectory(
            model, dataloader, opt, args.n_iters, r, d, args.clip_value,
            seed_traj=traj, traj_idx=traj, use_wandb=use_wandb,
        )
        results.append(
            dict(weights_init=w_init, weights_post=w_post, losses=losses, irelnorms=irelnorms)
        )

    config = vars(args).copy()
    config.update({"run_name": run_name, "d": d, "r": r, "n_layers": len(args.hiddens) + 1})

    print(f"\nSaving results:")
    save_results(run_name, args.output_dir, config, results)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

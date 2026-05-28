#!/usr/bin/env python3
"""
Train an MLP head on top of frozen CIFAR-style ResNet embeddings.

Goal
----
Real-data analogue of the synthetic support-selection experiments, but without
assuming known relevant/irrelevant coordinates.

We study whether different optimizers make the first MLP-head layer W1 collapse
onto a low-dimensional feature subspace.

Tracked metrics
---------------
For every checkpoint and every Linear layer:
  - weight matrix W
  - left Gram  W W^T
  - right Gram W^T W
  - singular values of W
  - eigenvalues of W W^T and W^T W
  - effective rank
  - stable rank
  - Frobenius norm
  - spectral norm
  - column norms and row norms

Optimizers
----------
  gd       : full-batch vanilla gradient descent
  sgd      : mini-batch SGD, batch size controlled by --batch_size
  gd_wd    : full-batch GD with weight decay, equivalent to --optimizer gd --weight_decay > 0
  adam     : Adam
  adamw    : AdamW
  muon     : local Muon implementation, for 2D matrices; 1D params fall back to SGD momentum

Typical usage
-------------
python train_cifar_resnet_head.py --optimizer gd --lr 0.01 --max_steps 2000 --run_name gd
python train_cifar_resnet_head.py --optimizer sgd --batch_size 128 --lr 0.01 --max_steps 2000 --run_name sgd128
python train_cifar_resnet_head.py --optimizer gd --lr 0.01 --weight_decay 0.1 --max_steps 2000 --run_name gd_wd01
python train_cifar_resnet_head.py --optimizer adam --lr 1e-3 --max_steps 2000 --run_name adam
python train_cifar_resnet_head.py --optimizer adamw --lr 1e-3 --weight_decay 0.01 --max_steps 2000 --run_name adamw
python train_cifar_resnet_head.py --optimizer muon --batch_size 128 --lr 0.02 --max_steps 2000 --run_name muon128

Notes
-----
By default the backbone is torchvision ResNet18 ImageNet weights. If you have a
CIFAR-trained ResNet checkpoint, pass --cifar_ckpt path/to/checkpoint.pth.
The script removes the final fc layer and caches 512-d embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim import Muon   


# -----------------------------------------------------------------------------
# Reproducibility / device
# -----------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# -----------------------------------------------------------------------------
# Backbone + CIFAR embeddings
# -----------------------------------------------------------------------------

def build_resnet18_backbone(cifar_ckpt: str | None) -> nn.Module:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    if cifar_ckpt is not None:
        state = torch.load(cifar_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            kk = k.removeprefix("module.").removeprefix("model.")
            if kk.startswith("fc."):
                continue
            cleaned[kk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded CIFAR checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def load_cifar10(data_dir: str, img_size: int):
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train, test


@torch.no_grad()
def extract_embeddings(backbone: nn.Module, dataset, batch_size: int, device: torch.device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    backbone = backbone.to(device)
    feats, labels = [], []
    for x, y in tqdm(loader, desc="extract embeddings", leave=False):
        z = backbone(x.to(device)).cpu()
        feats.append(z)
        labels.append(y)
    return torch.cat(feats).numpy().astype(np.float32), torch.cat(labels).numpy().astype(np.int64)


def get_or_create_embeddings(args, device: torch.device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "cifar" if args.cifar_ckpt else "imagenet"
    prefix = output_dir / f"embeddings_resnet18_{tag}_img{args.img_size}"
    paths = {
        "x_train": str(prefix) + "_train.npy",
        "y_train": str(prefix) + "_train_labels.npy",
        "x_test": str(prefix) + "_test.npy",
        "y_test": str(prefix) + "_test_labels.npy",
    }

    if not args.recompute_embeddings and all(Path(p).exists() for p in paths.values()):
        print("Loading cached embeddings ...")
        return tuple(np.load(paths[k]) for k in ["x_train", "y_train", "x_test", "y_test"])

    print("Extracting embeddings once and caching them ...")
    backbone = build_resnet18_backbone(args.cifar_ckpt)
    train_set, test_set = load_cifar10(args.data_dir, args.img_size)
    x_train, y_train = extract_embeddings(backbone, train_set, args.embed_batch_size, device)
    x_test, y_test = extract_embeddings(backbone, test_set, args.embed_batch_size, device)

    for key, arr in zip(paths.keys(), [x_train, y_train, x_test, y_test]):
        np.save(paths[key], arr)
    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------------
# MLP head
# -----------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int = 10, bias: bool = False):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout, bias=bias))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_weights(self) -> list[np.ndarray]:
        return [
            m.weight.detach().cpu().numpy().copy()
            for m in self.net.modules()
            if isinstance(m, nn.Linear)
        ]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def effective_rank_from_singular_values(s: np.ndarray, eps: float = 1e-12) -> float:
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def layer_metrics(W: np.ndarray, layer_idx: int) -> dict[str, float | np.ndarray]:
    s = np.linalg.svd(W, compute_uv=False)
    left_gram = W @ W.T
    right_gram = W.T @ W
    left_eigs = np.linalg.eigvalsh(left_gram)[::-1]
    right_eigs = np.linalg.eigvalsh(right_gram)[::-1]
    frob_sq = float(np.sum(s ** 2))
    spec_sq = float(s[0] ** 2 + 1e-12) if len(s) else 1e-12

    return {
        f"layer{layer_idx}_effective_rank": effective_rank_from_singular_values(s),
        f"layer{layer_idx}_stable_rank": frob_sq / spec_sq,
        f"layer{layer_idx}_frobenius_norm": float(np.sqrt(frob_sq)),
        f"layer{layer_idx}_spectral_norm": float(s[0]) if len(s) else 0.0,
        f"layer{layer_idx}_singular_values": s,
        f"layer{layer_idx}_left_gram_eigenvalues": left_eigs,
        f"layer{layer_idx}_right_gram_eigenvalues": right_eigs,
        f"layer{layer_idx}_column_norms": np.linalg.norm(W, axis=0),
        f"layer{layer_idx}_row_norms": np.linalg.norm(W, axis=1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y).long()
    total_loss, correct, n = 0.0, 0, 0
    for i in range(0, len(xt), batch_size):
        xb = xt[i:i + batch_size].to(device)
        yb = yt[i:i + batch_size].to(device)
        logits = model(xb)
        total_loss += float(crit(logits, yb).item())
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        n += len(xb)
    return total_loss / n, correct / n


def checkpoint_steps(max_steps: int, eval_every: int, checkpoint_fracs: list[float]) -> list[int]:
    steps = {0, max_steps}
    if eval_every > 0:
        steps.update(range(eval_every, max_steps + 1, eval_every))
    for f in checkpoint_fracs:
        steps.add(int(round(f * max_steps)))
    return sorted(s for s in steps if 0 <= s <= max_steps)


# -----------------------------------------------------------------------------
# Optimizer and training
# -----------------------------------------------------------------------------

def build_optimizer(args, params):
    if args.optimizer == "gd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "muon":
        return torch.optim.Muon(
            params,
            lr=args.lr,
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train_run(args, x_train, y_train, x_test, y_test, device):
    seed_all(args.seed)
    model = MLPHead(x_train.shape[1], args.hidden_dims, out_dim=10, bias=args.bias).to(device)
    opt = build_optimizer(args, model.parameters())
    crit = nn.CrossEntropyLoss()

    effective_bs = len(x_train) if args.optimizer == "gd" or args.batch_size == 0 else args.batch_size
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=effective_bs, shuffle=True, drop_last=False)
    data_iter = iter(loader)

    ckpts = set(checkpoint_steps(args.max_steps, args.eval_every, args.checkpoint_fracs))
    print(f"effective batch size = {effective_bs}; checkpoints = {len(ckpts)}")

    saved_weights: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float]] = []
    train_loss_steps: list[int] = []
    train_losses: list[float] = []

    def save_checkpoint(step: int):
        train_loss, train_acc = evaluate(model, x_train, y_train, args.eval_batch_size, device)
        test_loss, test_acc = evaluate(model, x_test, y_test, args.eval_batch_size, device)
        row: dict[str, float] = {
            "step": float(step),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        weights = model.linear_weights()
        for li, W in enumerate(weights):
            saved_weights[f"step{step}_layer{li}_weight"] = W
            ms = layer_metrics(W, li)
            for k, v in ms.items():
                if np.isscalar(v):
                    row[k] = float(v)
                else:
                    saved_weights[f"step{step}_{k}"] = np.asarray(v)

        metric_rows.append(row)
        ranks = ", ".join(
            f"L{li}:sr={row[f'layer{li}_stable_rank']:.1f},er={row[f'layer{li}_effective_rank']:.1f}"
            for li in range(len(weights))
        )
        print(f"step {step:>5}/{args.max_steps} | test_acc={100*test_acc:.2f}% | {ranks}")

    save_checkpoint(0)
    pbar = tqdm(range(1, args.max_steps + 1), desc=args.run_name)
    for step in pbar:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)

        model.train()
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xb), yb)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()

        train_loss_steps.append(step)
        train_losses.append(float(loss.item()))
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

        if step in ckpts:
            save_checkpoint(step)

    return saved_weights, metric_rows, np.asarray(train_loss_steps), np.asarray(train_losses)


# -----------------------------------------------------------------------------
# Save output
# -----------------------------------------------------------------------------

def save_npz_and_config(args, saved_weights, metric_rows, train_loss_steps, train_losses, embedding_dim: int):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name

    arrays: dict[str, np.ndarray] = dict(saved_weights)
    arrays["train_loss_steps"] = train_loss_steps
    arrays["train_losses"] = train_losses

    if metric_rows:
        keys = sorted(metric_rows[0].keys())
        for key in keys:
            arrays[f"metric_{key}"] = np.asarray([r[key] for r in metric_rows], dtype=np.float64)

    npz_path = out / f"{run_name}.npz"
    np.savez(npz_path, **arrays)

    config = vars(args).copy()
    config.update({
        "embedding_dim": embedding_dim,
        "n_layers": len(args.hidden_dims) + 1,
    })
    cfg_path = out / f"{run_name}_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved results: {npz_path}")
    print(f"saved config:  {cfg_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--optimizer", choices=["gd", "sgd", "adam", "adamw", "muon"], required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--max_steps", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=128, help="0 means full batch")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512])
    p.add_argument("--bias", action="store_true")

    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_nesterov", action="store_true")
    p.add_argument("--muon_ns_steps", type=int, default=5)

    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./results_clean_GD_25k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--embed_batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--cifar_ckpt", default=None, help="Optional CIFAR-trained ResNet18 checkpoint")
    p.add_argument("--recompute_embeddings", action="store_true")

    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--checkpoint_fracs", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if args.optimizer == "gd":
        args.batch_size = 0

    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"\n=== {args.run_name} ===")
    print(f"device={device} optimizer={args.optimizer} lr={args.lr} wd={args.weight_decay} max_steps={args.max_steps}")

    x_train, y_train, x_test, y_test = get_or_create_embeddings(args, device)
    print(f"embeddings: train={x_train.shape}, test={x_test.shape}")

    saved_weights, metric_rows, train_loss_steps, train_losses = train_run(
        args, x_train, y_train, x_test, y_test, device
    )
    save_npz_and_config(args, saved_weights, metric_rows, train_loss_steps, train_losses, x_train.shape[1])


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyparsing

os.makedirs("generated", exist_ok=True)

# Configuration

ORDER = [
    "GD wd=0.1",
    "GD",
    "SGD bs=512",
    "SGD bs=32",
    "ADAM bs=512",
    "MUON bs=512",
]

# Unused / experimental functions (kept for reference)

def effective_rank(W: np.ndarray, eps: float = 1e-12) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    p = s / (s.sum() + eps)
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))

def plot_effective_rank(runs, target, save=True):
    labels = ["Init"] + [r["label"] for r in runs]
    ranks  = [effective_rank(runs[0]["W1_init"])] + \
             [effective_rank(r["W1_post"]) for r in runs]
    fig, ax = plt.subplots(figsize=(len(labels) * 1.2, 4))
    ax.bar(labels, ranks, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axhline(runs[0]["cfg"]["r"], color="black", linestyle="--",
               linewidth=1.2, label=f"r = {runs[0]['cfg']['r']}")
    ax.set_ylabel("effective rank", fontsize=10)
    ax.set_title("Effective rank of W1 - " + target, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(f"generated/effective_rank_{target}.png", dpi=150, bbox_inches="tight")
    plt.show()

def stable_rank(W):
    svs = np.linalg.svd(W, compute_uv=False)
    return (svs**2).sum() / (svs[0]**2 + 1e-12)

def plot_stable_rank(runs, target, save=True):
    labels = ["Init"] + [r["label"] for r in runs]
    ranks  = [stable_rank(runs[0]["W1_init"])] + \
             [stable_rank(r["W1_post"]) for r in runs]
    fig, ax = plt.subplots(figsize=(len(labels) * 1.2, 4))
    ax.bar(labels, ranks, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("stable rank", fontsize=10)
    ax.set_title("Stable rank of W1 - " + target, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(f"generated/stable_rank_{target}.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_eigenvalues(runs, target, save=True):
    n_plots = len(runs) + 1
    fig, axs = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 4))
    def _plot_one(ax, W, title, loss=None):
        eigs = np.linalg.eigvalsh(W)
        ax.hist(eigs, bins=15, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{title}  ({loss:.4f})" if loss is not None else title, fontsize=9)
        ax.set_xlabel("eigenvalue", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("frequency", fontsize=9)
    _plot_one(axs[0], runs[0]["W1_init"], "Initialization")
    for i, r in enumerate(runs):
        _plot_one(axs[i + 1], r["W1_post"], r["label"], loss=r["losses"][-1])
    fig.suptitle("Eigenvalue histogram of W1 - " + target, fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(f"generated/eigenvalues_{target}.png", dpi=150, bbox_inches="tight")
    plt.show()


TRAJ_IDX  = 0
LAYER_IDX = 0

COLORS = {
    "GD":           "steelblue",
    "SGD bs=512":   "orange",
    "SGD bs=32":    "green",
    "ADAM bs=512":  "red",
    "MUON bs=512":  "purple",
    "GD wd=0.1":    "deeppink",
}

def make_label(cfg):
    opt = cfg["optimizer"].upper()
    wd  = f" wd={cfg['weight_decay']}" if cfg["weight_decay"] != 0.0 else ""
    bs  = f" bs={cfg['batch_size']}"   if cfg["optimizer"] != "gd"    else ""
    return f"{opt}{bs}{wd}"

def gram(W):
    return np.abs(W.T @ W)

def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def load_runs(output_dir, traj_idx=TRAJ_IDX, layer_idx=LAYER_IDX):
    runs_dict = {}
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".npz"):
            continue
        run_name = fname[:-4]
        npz_path = os.path.join(output_dir, fname)
        cfg_path = os.path.join(output_dir, f"{run_name}_config.json")
        data = np.load(npz_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg["r"] != 5:
            continue
        label = make_label(cfg)
        if label not in ORDER or label in runs_dict:
            continue
        t, l = traj_idx, layer_idx
        runs_dict[label] = {
            "label":     label,
            "run_name":  run_name,
            "cfg":       cfg,
            "W1_init":   data[f"traj{t}_layer{l}_init"],
            "W1_post":   data[f"traj{t}_layer{l}_post"],
            "losses":    data[f"traj{t}_losses"],
            "irelnorms": data[f"traj{t}_irelnorms"],
        }
    return [runs_dict[label] for label in ORDER if label in runs_dict]

def print_metrics(runs):
    print(f"  {'Label':25s}  {'loss':>8s}  {'irel_init':>10s}  {'irel_final':>10s}")
    print("  " + "-" * 62)
    for r in runs:
        print(f"  {r['label']:25s}  loss={r['losses'][-1]:.4f}  irelnorm_init={r['irelnorms'][0]:.3f}  irelnorm_final={r['irelnorms'][-1]:.3f}")

def plot_w1_gram(runs, target, save=True):
    n = len(runs)
    fig, axs = plt.subplots(2, n + 1, figsize=(3 * (n + 1), 7))
    if n + 1 == 1:
        axs = axs[:, np.newaxis]

    W1_init = runs[0]["W1_init"]
    gram_init = minmax(gram(W1_init) ** 0.95)
    axs[0, 0].imshow(np.abs(W1_init), cmap="viridis", interpolation="nearest")
    axs[0, 0].set_title("Initialization", fontsize=14)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[1, 0].imshow(gram_init, cmap="magma", interpolation="nearest")
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    grams_post = [minmax(gram(r["W1_post"]) ** 0.95) for r in runs]
    vmin = min(g.min() for g in grams_post)
    vmax = max(g.max() for g in grams_post)
    for i, r in enumerate(runs):
        axs[0, i + 1].imshow(np.abs(r["W1_post"]), cmap="viridis", interpolation="nearest")
        axs[0, i + 1].set_title(r["label"], fontsize=14)
        axs[0, i + 1].set_xticks([])
        axs[0, i + 1].set_yticks([])
        axs[1, i + 1].imshow(grams_post[i], cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[1, i + 1].set_xticks([])
        axs[1, i + 1].set_yticks([])

    axs[0, 0].set_ylabel("W1", fontsize=14)
    axs[1, 0].set_ylabel("Gram(W1)", fontsize=14)
    fig.suptitle(f"W1 and Gram(W1) - {target} - post-training", fontsize=16)
    plt.tight_layout()
    if save:
        path = f"generated/w1_gram_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

def plot_w1_only(runs, target, save=True):
    n = len(runs)
    fig, axs = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.5))
    if n + 1 == 1:
        axs = [axs]

    axs[0].imshow(np.abs(runs[0]["W1_init"]), cmap="viridis", interpolation="nearest")
    axs[0].set_title("Initialization", fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_ylabel("W1", fontsize=11)

    for i, r in enumerate(runs):
        axs[i + 1].imshow(np.abs(r["W1_post"]), cmap="viridis", interpolation="nearest")
        axs[i + 1].set_title(r["label"], fontsize=14)
        axs[i + 1].set_xticks([])
        axs[i + 1].set_yticks([])

    fig.suptitle(f"W1 - {target} - post-training", fontsize=16)
    plt.tight_layout()
    if save:
        path = f"generated/w1_only_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

def plot_irelnorm_only(runs, target, save=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    custom_order = ["ADAM bs=512", "MUON bs=512", "GD wd=0.1", "SGD bs=32", "SGD bs=512", "GD"]
    ordered = sorted(runs, key=lambda r: custom_order.index(r["label"]) if r["label"] in custom_order else 99)
    for r in ordered:
        c = COLORS.get(r["label"], "gray")
        ax.semilogx(r["irelnorms"], label=r["label"], color=c, linewidth=1.0)
    ax.set_title(r"Irelnorm  $\|W_1[:, r:]\|$" + f" - {target}")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save:
        path = f"generated/irelnorm_only_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

def plot_training_curves(runs, target, save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    custom_order = ["ADAM bs=512", "MUON bs=512", "GD wd=0.1", "SGD bs=32", "SGD bs=512", "GD"]
    ordered = sorted(runs, key=lambda r: custom_order.index(r["label"]) if r["label"] in custom_order else 99)
    for r in ordered:
        c = COLORS.get(r["label"], "gray")
        ax1.loglog(r["losses"],    label=r["label"], color=c, linewidth=1.0)
        ax2.semilogx(r["irelnorms"], label=r["label"], color=c, linewidth=1.0)
    ax1.set_title(f"Loss (MSE) - {target}")
    ax1.set_xlabel("iteration")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)
    ax2.set_title(r"Irelnorm  $\|W_1[:, r:]\|$" + f" - {target}")
    ax2.set_xlabel("iteration")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save:
        path = f"generated/training_curve_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

def plot_sv_spectrum(runs, target, save=True):
    fig, axs = plt.subplots(1, len(runs) + 1, figsize=(4 * (len(runs) + 1), 4),
                            sharey=True, sharex=True)

    def _plot_one(ax, W, title, loss=None):
        svs = np.linalg.svd(W, compute_uv=False)
        ax.bar(range(len(svs)), svs, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{title}  ({loss:.4f})" if loss is not None else title, fontsize=11)
        ax.set_xlabel("singular value index", fontsize=8)
        ax.set_xticks(range(len(svs)))
        ax.set_xticklabels(range(len(svs)), fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    axs[0].set_ylabel("magnitude", fontsize=9)
    _plot_one(axs[0], runs[0]["W1_init"], "Initialization")
    for i, r in enumerate(runs):
        _plot_one(axs[i + 1], r["W1_post"], r["label"], loss=r["losses"][-1])

    fig.suptitle("Singular value spectrum of W1 - " + target, fontsize=12)
    plt.tight_layout()
    if save:
        path = f"generated/sv_spectrum_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

def plot_irelnorm_ratio(runs, target, save=True):
    labels = ["Init"] + [r["label"] for r in runs]
    ratios = [
        np.linalg.norm(runs[0]["W1_init"][:, runs[0]["cfg"]["r"]:]) /
        (np.linalg.norm(runs[0]["W1_init"]) + 1e-12)
    ] + [
        np.linalg.norm(r["W1_post"][:, r["cfg"]["r"]:]) /
        (np.linalg.norm(r["W1_post"]) + 1e-12)
        for r in runs
    ]

    fig, ax = plt.subplots(figsize=(len(labels) * 1.2, 4))
    ax.bar(labels, ratios, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("||W1[:,r:]|| / ||W1||", fontsize=10)
    ax.set_title("Irrelevant norm ratio - " + target, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    if save:
        path = f"generated/irelnorm_ratio_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()

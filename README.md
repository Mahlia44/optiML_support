# Which Optimizers Find the Support?
## Optimization for Machine Learning - CS-439
### by Mahlia Merville-Hipeau, François Goybet, Matteo Pinto

A study of whether different optimizers (GD, SGD, Adam, Muon) learn to
identify the **relevant feature support** of a sparse target function - i.e.
whether the first layer of an MLP collapses onto the few coordinates that
actually matter and discards the irrelevant ones.

The setup follows the sparse-support learning problem described by Beneventano et al. in
[How Neural Networks Learn the Support is an Implicit Regularization Effect of SGD](https://arxiv.org/abs/2406.11110). We train small MLPs on
synthetic targets where only the first `r` of `d` input features are relevant,
and measure how much weight each optimizer places on the irrelevant
coordinates. A preliminary CIFAR-10 extension is included as a realistic (and
harder to interpret) analogue.

The full write-up is in [report.pdf](report.pdf).

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync
the environment:

```bash
uv sync
```

Training curves are optionally logged to [Weights & Biases](https://wandb.ai).
To enable logging, log in once and pass your entity at run time
(or use `--no_wandb` to skip it entirely):

```bash
uv run wandb login
```

## Repository structure

```
optiML_support/
├── train_baseline.py      # train MLPs on synthetic targets (main entry point)
├── visualize.ipynb        # generate all report plots from saved runs
├── source/                # model and plotting utilities
│   ├── models.py          # configurable linear / ReLU MLP
│   ├── trainer.py         # training loop + per-layer metric callbacks
│   └── utils.py           # plotting and analysis helpers
├── outputs/               # saved runs (the trained "models")
│   ├── linear/            # one folder per target function, each holding
│   ├── sine/              #   {run}.npz        – weights (init/post) + curves
│   └── staircase/         #   {run}_config.json – full hyperparameter config
├── cifar-experiments/     # realistic CIFAR-10 extension (see its own README)
└── report.pdf             # report
```

- **`outputs/`** stores the trained models. Each run produces an `.npz` file
  (initial/final weight matrices plus loss and irrelevant-norm curves) and a
  matching `_config.json`. These are the inputs consumed by the notebook.
- **`source/`** holds the reusable code: the `MLP` definition and the plotting/analysis utilities.
- **`cifar-experiments/`** contains a self-contained extension that trains an
  MLP head on frozen ResNet-18 embeddings. See
  [cifar-experiments/README.md](cifar-experiments/README.md) for details.

## Training

`train_baseline.py` trains an MLP on a synthetic dataset and writes the weights
and training curves to `--output_dir`.

**Key arguments**

| Argument          | Choices / type                          | Description                              |
|-------------------|-----------------------------------------|------------------------------------------|
| `--optimizer`     | `gd`, `sgd`, `adam`, `adamw`, `muon`    | `gd` is full-batch (ignores batch size)  |
| `--target`        | `linear`, `sine`, `staircase`           | Synthetic target function                |
| `--batch_size`    | int                                     | Mini-batch size (SGD / Adam / Muon)      |
| `--n_iters`       | int                                     | Total gradient steps                     |
| `--lr`            | float                                   | Learning rate                            |
| `--weight_decay`  | float                                   | L2 weight decay                          |
| `--n_trajs`       | int                                     | Independent training trajectories        |
| `--output_dir`    | path                                    | Where to write results                   |

Run `uv run train_baseline.py --help` for the full list.

**Example run** — full-batch GD on the linear target:

```bash
uv run train_baseline.py \
    --optimizer gd \
    --target linear \
    --n_iters 200000 \
    --output_dir outputs/linear
```

Swap `--optimizer`, `--target`, and `--batch_size` to reproduce the other runs
in `outputs/`. A quick smoke test:

```bash
uv run train_baseline.py --optimizer gd --target linear --n_iters 50 --output_dir outputs/test
```

## Visualization

Once runs are saved under `outputs/`, open
[visualize.ipynb](visualize.ipynb) to reproduce every figure in the report
(loss curves, irrelevant-weight norms, and per-target comparisons across
optimizers):

```bash
uv run jupyter notebook visualize.ipynb
```

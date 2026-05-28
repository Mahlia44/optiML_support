# Optimization for ML
# SGD finds support

https://arxiv.org/pdf/2406.11110


- Muon try other params -> Mahlia
- staircase -> Mahlia
- check for CIFAR10 implementation -> Matteo
- try to understand why Adam works -> François
- start analysis and discussion: François

- metrics:
    - weight matrix
    - gram matrix
    - bar plot (eigenvalues, singularvalues)
    - norm irrelevant
    - per layer effective rank

Report :
- read guidelines
- describe the goal 
- comparisons :
    - optimizers
    - target functions


## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

Training curves are logged to [Weights & Biases](https://wandb.ai). Make sure you are logged in:

```bash
uv run wandb login
```

The script will prompt you for your **wandb entity** (your W&B username or team name) at the start of each run. You can also pass it directly with `--wandb_entity <name>` to skip the prompt. Use `--no_wandb` to disable logging entirely.

## Training baselines

`train_baseline.py` trains an MLP on a synthetic dataset and saves weights + curves to disk for visualization.

**Key arguments**
- `--optimizer`: `gd` (full-batch), `sgd`, `adam`, `adamw`, `muon`
- `--batch_size`: mini-batch size (SGD/Adam/Muon only)
- `--target`: `linear`, `sine`, `staircase`
- `--n_iters`, `--lr`, `--weight_decay`, `--n_trajs`, `--hiddens`, ...

Outputs land in `--output_dir` as `{run_name}.npz` (weights + curves) and `{run_name}_config.json`.

**Smoke test**
```bash
uv run train_baseline.py --optimizer gd --target linear --n_iters 50 --output_dir outputs/test
```

**Full runs**
Linear
```bash
uv run train_baseline.py --optimizer gd   --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 512 --target linear --n_iters 200000 --n_trajs 5 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 32  --target linear --n_iters 200000 --n_trajs 5 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer gd   --weight_decay 0.1 --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer muon --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer adam --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer adamw --weight_decay 0.01 --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
```

Sine
```bash
uv run train_baseline.py --optimizer gd   --target sine --n_iters 200000 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 512 --target sine --n_iters 200000 --n_trajs 5 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 32  --target sine --n_iters 200000 --n_trajs 5 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer gd   --weight_decay 0.1 --target sine --n_iters 200000 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer muon --target sine --n_iters 200000 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer adam --target sine --n_iters 200000 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer adamw --weight_decay 0.01 --target sine --n_iters 200000 --output_dir outputs/sine --wandb_entity mahlia-merville-epfl
```









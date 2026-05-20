# sgd-finds-support

https://arxiv.org/pdf/2406.11110


optimizers à tester: 
Adam,AdamW,Muon,Frank-Wolfe

params pour le .py unifié: optimizers, target functions & hyperparams

tous sur linear synthetic dataset
Mahlia: run baseline sur SGD&friends
François: clone baseline et fait sur Muon
Matteo: clone baseline et fait sur Adam
Mahlia,François,Matteo: have a lot of fun :)



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
```bash
uv run train_baseline.py --optimizer gd   --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 512 --target linear --n_iters 200000 --n_trajs 5 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer sgd  --batch_size 32  --target linear --n_iters 200000 --n_trajs 5 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer gd   --weight_decay 0.1 --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer muon --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl

uv run train_baseline.py --optimizer adam --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
uv run train_baseline.py --optimizer adamw --target linear --n_iters 200000 --output_dir outputs/linear --wandb_entity mahlia-merville-epfl
```
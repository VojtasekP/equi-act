# multi_run.py
import os, random, argparse
import numpy as np
import torch
import wandb
import yaml
from types import SimpleNamespace

from train import _train_impl  # must be the patched version that RETURNS metrics

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config with hyperparams")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed0", type=int, default=1337)
    p.add_argument("--wandb_group", type=str, default="multi-run")
    p.add_argument("--wandb_mode", type=str, default=os.getenv("WANDB_MODE", "online"))
    args_cli = p.parse_args()

    # 1) Load YAML (hyperparams live here)
    with open(args_cli.config, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    # 2) Merge run controls (not model hparams)
    runs = int(args_cli.runs)
    seed0 = int(args_cli.seed0)
    wandb_group = args_cli.wandb_group
    os.environ["WANDB_MODE"] = args_cli.wandb_mode  # "online" or "offline"

    # 3) Make a mutable, attribute-style config object from YAML
    #    Ensure lists have correct types (e.g., channels_per_block)
    if "channels_per_block" in cfg_dict and isinstance(cfg_dict["channels_per_block"], list):
        cfg_dict["channels_per_block"] = [int(x) for x in cfg_dict["channels_per_block"]]

    base = SimpleNamespace(**cfg_dict)

    # Required minimal keys sanity
    for k in ["project", "name", "dataset", "epochs", "batch_size", "lr", "weight_decay"]:
        if not hasattr(base, k):
            raise ValueError(f"Missing key `{k}` in YAML config.")

    accs, losses, paths, best_vals = [], [], [], []

    for i in range(runs):
        seed = seed0 + i
        set_seed(seed)

        # Per-run name (keeps base.name prefix)
        run_name = f"{base.name}-seed{seed}"

        # Start a W&B run (not a sweep)
        wandb.init(project=base.project, name=run_name, group=wandb_group, reinit=True)
        # Log full config + seed for reproducibility
        wandb.config.update({**cfg_dict, "seed": seed}, allow_val_change=True)

        # Build a fresh config object per run (copy base, override name if you want)
        cfg = SimpleNamespace(**{**cfg_dict, "name": run_name})

        try:
            out = _train_impl(cfg)  # returns dict with: test_acc, test_loss, best_val_acc, best_acc_ckpt, ...
        finally:
            wandb.finish()

        losses.append(out["test_loss"])
        best_vals.append(out.get("best_val_acc"))
        paths.append(out["best_ckpt"])

        print(f"[seed={seed}] test_acc={out['test_acc']:.4f} | best_val_acc={out.get('best_val_acc', float('nan')):.4f}}")

    # 4) Aggregate results
    accs_np = np.array(accs, dtype=float)
    mean_acc = float(np.nanmean(accs_np))
    std_acc = float(np.nanstd(accs_np, ddof=1)) if len(accs_np) > 1 else 0.0

    print("\n=== Summary over {} runs ===".format(runs))
    print(f"Test accuracy: mean={mean_acc:.4f}, std={std_acc:.4f}")
    print("Best-ACC checkpoints per run:")
    for pth in paths:
        print("  ", pth)

if __name__ == "__main__":
    main()

# sweeps/train_grid.py
import argparse, csv, os, random
from datetime import datetime
from types import SimpleNamespace
import torch

from SO2_Nets.train import train  # uses Lightning + W&B internally



ACTIVATIONS = [
    "gated_relu", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu",
]

NORMALIZATIONS = [
    "IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm",
]

BASE_CFG = dict(
    project="equi-act-grid",
    name="grid",
    epochs=12,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    channels_per_block=[16, 32, 64],
    layers_per_block=2,
    kernel_size=5,
    pool_stride=2,
    pool_sigma=0.9,
    invariant_channels=128,
    max_rot_order=3,
    img_size=128,
    n_classes=10,
    patience=6,
)

HEADER = [
    "timestamp","dataset","activation","normalization","repeat_idx","seed",
    "epochs","batch_size","lr","weight_decay","channels_per_block",
    "layers_per_block","kernel_size","pool_stride","pool_sigma",
    "invariant_channels","max_rot_order","img_size",
    "best_val_loss","test_acc","test_loss","best_ckpt",
    "device","cuda_name","status",
]

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_one(cfg_dict):
    cfg = SimpleNamespace(**cfg_dict)
    return train(cfg)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mnist_rot","resisc45","colorectal_hist"])
    p.add_argument("--repeats", type=int, default=3, help="n runs per (activation, bn)")
    p.add_argument("--seed0", type=int, default=0, help="base seed; run i uses seed0+i")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--channels_per_block", default=[16, 32, 64])
    p.add_argument("--layers_per_block", type=int, default=2)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--pool_stride", type=int, default=2)
    p.add_argument("--pool_sigma", type=float, default=0.9)
    p.add_argument("--invariant_channels", type=int, default=128)
    p.add_argument("--max_rot_order", type=int, default=3)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--project", type=str, default="equi-act-grid")
    p.add_argument("--out", type=str, default="sweeps/results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv = os.path.join(args.out, f"runs_{args.dataset}_{ts}.csv")

    channels_per_block = args.channels_per_block

    base = BASE_CFG.copy()
    base.update(dict(
        project=args.project,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        channels_per_block=channels_per_block,
        layers_per_block=args.layers_per_block,
        kernel_size=args.kernel_size,
        pool_stride=args.pool_stride,
        pool_sigma=args.pool_sigma,
        invariant_channels=args.invariant_channels,
        max_rot_order=args.max_rot_order,
        img_size=args.img_size
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    with open(runs_csv, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

    total = len(ACTIVATIONS) * len(NORMALIZATIONS) * args.repeats
    k = 0
    for activation in ACTIVATIONS:
        for bn in NORMALIZATIONS:
            for r in range(args.repeats):
                k += 1
                seed = args.seed0 + r
                set_seed(seed)
                cfg = base.copy()
                cfg.update(dict(
                    dataset=args.dataset,
                    activation_type=activation,
                    bn=bn,
                    name=f"{args.dataset}-{activation}-{bn}-rep{r}-seed{seed}",
                ))
                status = "OK"
                best_val_loss = test_acc = test_loss = ""
                best_ckpt = ""
                try:
                    print(f"[{k}/{total}] {args.dataset} | {activation}+{bn} | rep={r} seed={seed}")
                    res = run_one(cfg)
                    best_val_loss = f'{res.get("best_val_loss"):.6f}' if res.get("best_val_loss") is not None else ""
                    test_acc = f'{res.get("test_acc"):.6f}' if res.get("test_acc") is not None else ""
                    test_loss = f'{res.get("test_loss"):.6f}' if res.get("test_loss") is not None else ""
                    best_ckpt = res.get("best_ckpt","")
                except Exception as e:
                    status = f"ERROR: {type(e).__name__}: {e}"

                row = [
                    datetime.now().isoformat(timespec="seconds"),
                    args.dataset, activation, bn, r, seed,
                    cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], str(cfg["channels_per_block"]),
                    cfg["layers_per_block"], cfg["kernel_size"], cfg["pool_stride"], cfg["pool_sigma"],
                    cfg["invariant_channels"], cfg["max_rot_order"], cfg["img_size"],
                    best_val_loss, test_acc, test_loss, best_ckpt,
                    device, cuda_name, status,
                ]
                with open(runs_csv, "a", newline="") as f:
                    csv.writer(f).writerow(row)

    print(f"Runs CSV: {runs_csv}")

if __name__ == "__main__":
    main()

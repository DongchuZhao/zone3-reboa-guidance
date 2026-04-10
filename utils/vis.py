# utils/vis.py
from __future__ import annotations
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def write_csv(path: str, header, rows):
    ensure_dir(os.path.dirname(path))
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        w.writerow(rows)

def plot_curves(history: dict, out_png: str, title: str):
    """
    history: dict(key -> list of scalars)
    [internal]
    """
    ensure_dir(os.path.dirname(out_png))
    ep = range(1, len(history["epoch"])+1)

    plt.figure(figsize=(12,8))
    # 1) MSE(norm)
    if "train_mse" in history and "val_mse" in history:
        plt.subplot(3,1,1)
        plt.plot(ep, history["train_mse"], label="train MSE(norm)")
        plt.plot(ep, history["val_mse"],   label="val MSE(norm)")
        plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)

    # 2) RMSE(mm)
    keys2 = ["coord_rmse_mm","diam_rmse_mm","len_rmse_mm"]
    if all("val_"+k in history for k in keys2):
        plt.subplot(3,1,2)
        for k in keys2:
            plt.plot(ep, history["val_"+k], label="val "+k)
        plt.xlabel("epoch"); plt.ylabel("RMSE (mm)"); plt.legend(); plt.grid(True)

    # 3) MAE(mm)
    keys3 = ["coord_mae_mm","diam_mae_mm","len_mae_mm"]
    if all("val_"+k in history for k in keys3):
        plt.subplot(3,1,3)
        for k in keys3:
            plt.plot(ep, history["val_"+k], label="val "+k)
        plt.xlabel("epoch"); plt.ylabel("MAE (mm)"); plt.legend(); plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_png, dpi=160)
    plt.close()

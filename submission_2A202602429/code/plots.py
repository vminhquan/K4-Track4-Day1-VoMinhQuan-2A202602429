"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png)."""
from __future__ import annotations

import matplotlib.pyplot as plt

LABELS = {"train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
          "val_macro_f1": "val macro-F1", "grad_norm": "grad_norm (trước clip, TB epoch)"}


def _cfg_text(cfg: dict) -> str:
    parts = [f"loss={cfg['loss']}", f"opt={cfg['optimizer']}", f"lr={cfg['lr']:g}",
             f"wd={cfg['weight_decay']:g}", f"batch={cfg['batch']}", f"hidden={tuple(cfg['hidden'])}",
             f"drop={cfg['dropout']}", f"clip={cfg['clip_norm']}", f"{cfg['precision']}",
             f"init={cfg['init']}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip (TB và max mỗi epoch)."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label=LABELS["train_loss"])
    ax.plot(ep, h["val_loss"], "s-", ms=3, label=LABELS["val_loss"])
    ax.set(title=f"Loss ({cfg['loss']})", xlabel="epoch", ylabel="loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label=LABELS["val_acc"])
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label=LABELS["val_macro_f1"])
    ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán đa số (acc 0.4876)")
    ax.set(title="Val accuracy / macro-F1", xlabel="epoch", ylabel="score")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="grad_norm TB")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="grad_norm max")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls="--", lw=1, label=f"c = {cfg['clip_norm']:g}")
    ax.set(title="Chuẩn gradient (trước clip)", xlabel="epoch", ylabel="‖g‖₂", yscale="log")

    for ax in axes:
        if s["best_epoch"] > 0:
            ax.axvline(s["best_epoch"], color="green", ls=":", lw=1, label=f"best epoch {s['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    status = "  [DIVERGED]" if s["diverged"] else ""
    fig.suptitle(f"{cfg['exp_id']}{status} — {_cfg_text(cfg)}\n"
                 f"best val_loss={s['best_val_loss']:.4f} @ep{s['best_epoch']}, "
                 f"val_acc={s['val_acc']:.4f}, val_macro_f1={s['val_macro_f1']:.4f}, "
                 f"step0={s['step0_loss']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số (mỗi chỉ số một ô) của nhiều thí nghiệm, chú thích bằng exp_id."""
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=2.5, label=r["cfg"]["exp_id"])
        ax.set(title=LABELS.get(m, m), xlabel="epoch", ylabel=m)
        if m == "grad_norm":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title or f"So sánh: {', '.join(metrics)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

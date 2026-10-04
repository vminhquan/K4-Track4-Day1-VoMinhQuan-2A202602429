"""optimizer.py — chọn bộ tối ưu, lập lịch lr và cắt gradient.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam là L2 cộng vào gradient (bị chia bởi sqrt(v_hat)),
    còn của AdamW là suy giảm tách riêng (không bị chia).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận '{name}'")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, warmup_steps: int = 0, **kwargs):
    """None | "cosine" (cosine annealing theo BƯỚC, tuỳ chọn khởi động tuyến tính warmup_steps bước).

    Trả về LambdaLR, gọi scheduler.step() sau mỗi optimizer.step().
    """
    if name is None:
        return None
    if name != "cosine":
        raise ValueError(f"scheduler chưa hỗ trợ: {name}")

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    """Cắt gradient theo chuẩn L2 toàn cục và TRẢ VỀ chuẩn TRƯỚC KHI cắt.

    max_norm=None: chỉ đo (clip_grad_norm_ với max_norm=inf không thay đổi gradient).
    Với FP16 + GradScaler phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    Trả về tensor 0 chiều (còn trên GPU) thay vì float để không đồng bộ GPU ở mỗi bước;
    train.py gom lại và chỉ chuyển về CPU một lần cuối epoch.
    """
    params = [p for p in params if p.grad is not None]
    limit = float("inf") if max_norm is None else max_norm
    return torch.nn.utils.clip_grad_norm_(params, limit).detach()

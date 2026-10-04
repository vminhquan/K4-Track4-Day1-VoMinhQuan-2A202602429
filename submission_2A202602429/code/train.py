"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2) rồi ghi đè.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine"
    seed=1,
    train_eval_size=50_000,    # tập con CỐ ĐỊNH của train để đo train_loss ở chế độ eval()
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(y_true, y_pred, k: int = N_CLASSES) -> np.ndarray:
    """Hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py)."""
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_scores(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, chế độ eval()."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : F.cross_entropy(logit thô, nhãn int64).
       "mse" : F.mse_loss(logits, one_hot(y)) — áp lên LOGIT (không softmax), trung bình trên MỌI phần tử
               (B * 7), không có hệ số 1/2. Với reduction="sum" thì cộng trên lô rồi chia 7 để mỗi mẫu
               đóng góp đúng bằng giá trị "mean" theo mẫu.
    """
    logits = logits.float()  # tính loss ở FP32 kể cả khi autocast
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, N_CLASSES).float()
        if reduction == "sum":
            return F.mse_loss(logits, target, reduction="sum") / N_CLASSES
        return F.mse_loss(logits, target, reduction="mean")
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận '{loss_name}'")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt), FP32."""
    model.eval()
    total, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds).cpu().numpy()
    yt = y.cpu().numpy()
    cm = confusion_matrix(yt, pred)
    return dict(loss=total / len(X), acc=float((pred == yt).mean()), macro_f1=macro_f1_from_confusion(cm))


def build_model(cfg: dict, device) -> MLP:
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    assert count_params(model) == EXPECTED_PARAMS[hidden], \
        f"{hidden}: {count_params(model)} tham số, cần {EXPECTED_PARAMS[hidden]}"
    return model.to(device)


def _autocast(precision: str, device_type: str):
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - train_loss đo ở chế độ eval() trên một tập con CỐ ĐỊNH (cfg["train_eval_size"] mẫu đầu của X_tr,
      vốn đã được xáo bởi train_test_split) để so sánh được với val_loss.
    - grad_norm: chuẩn L2 toàn cục TRƯỚC khi clip, trung bình (và max) trong epoch.
    - best_epoch = epoch có val_loss thấp nhất; summary lấy val_acc/val_macro_f1 ở epoch đó.
    - Loss không hữu hạn -> diverged=True, dừng sớm.
    TUYỆT ĐỐI không dùng X_eval ở đây.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "cfg['lr'] chưa được đặt"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev_type = device.type
    n_sub = min(cfg["train_eval_size"], len(X_tr))
    X_sub, y_sub = X_tr[:n_sub], y_tr[:n_sub]

    # 0. seed, model, optimizer
    set_seed(cfg["seed"])
    model = build_model(cfg, device)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    sched = build_scheduler(opt, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"])
    use_scaler = cfg["precision"] == "fp16" and dev_type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    gen = torch.Generator().manual_seed(cfg["seed"])  # thứ tự lô chỉ phụ thuộc seed
    if dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. loss bước 0, TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7 với CE)
    step0 = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best_val, best_epoch, best_state, diverged = float("inf"), 0, None, False

    # 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        norms, losses = [], []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with _autocast(cfg["precision"], dev_type):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)  # bỏ hệ số s TRƯỚC khi đo/cắt gradient
            else:
                loss.backward()
            norms.append(clip_gradients(model.parameters(), cfg["clip_norm"]))
            losses.append(loss.detach())
            if scaler is not None:
                scaler.step(opt)  # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                opt.step()
            if sched is not None:
                sched.step()
        _sync(device)
        ep_time = time.perf_counter() - t0

        norms_t = torch.stack(norms).float().cpu()
        loss_t = torch.stack(losses).float().cpu()
        finite_norms = norms_t[torch.isfinite(norms_t)]  # FP16: vài bước overflow bị scaler bỏ qua
        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(finite_norms.mean()) if len(finite_norms) else float("nan"))
        hist["grad_norm_max"].append(float(finite_norms.max()) if len(finite_norms) else float("nan"))
        hist["clip_frac"].append(float((norms_t > cfg["clip_norm"]).float().mean())
                                 if cfg["clip_norm"] is not None else 0.0)
        hist["epoch_time_s"].append(ep_time)

        if verbose:
            print(f"[{cfg['exp_id']}] ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"acc {va['acc']:.4f}  f1 {va['macro_f1']:.4f}  gn {hist['grad_norm'][-1]:.3f}  {ep_time:.1f}s")

        # loss huấn luyện không hữu hạn ở bất kỳ bước nào (với FP16, scaler đã xử lý overflow của gradient,
        # nhưng loss forward vẫn phải hữu hạn), hoặc val loss không hữu hạn -> phân kỳ
        if not torch.isfinite(loss_t).all() or not math.isfinite(va["loss"]):
            diverged = True
            if verbose:
                print(f"[{cfg['exp_id']}] DIVERGED ở epoch {epoch}, dừng sớm")
            break
        if va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 3. tóm tắt tại best_epoch
    if best_epoch > 0:
        bi = best_epoch - 1
        best_acc, best_f1 = hist["val_acc"][bi], hist["val_macro_f1"][bi]
    else:  # phân kỳ ngay epoch 1
        best_acc = best_f1 = float("nan")
    summary = dict(
        step0_loss=step0,
        best_val_loss=best_val if best_epoch > 0 else float("nan"),
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1],
        final_val_loss=hist["val_loss"][-1],
        val_acc=best_acc,
        val_macro_f1=best_f1,
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=(torch.cuda.max_memory_allocated() / 2**20) if dev_type == "cuda" else float("nan"),
        diverged=diverged,
        steps_per_epoch=steps_per_epoch,
    )
    return dict(cfg=copy.deepcopy(cfg), history=hist, summary=summary, best_state=best_state)


def write_predictions(row_id, preds, path: str) -> None:
    """CSV `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng eval, mỗi row_id đúng một lần."""
    import pandas as pd
    row_id, preds = np.asarray(row_id).astype(np.int64), np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng cho baseline và cấu hình cuối: nạp best_state, dự đoán eval (fp32, eval mode), ghi CSV."""
    model = build_model({**DEFAULT_CFG, **cfg}, data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds

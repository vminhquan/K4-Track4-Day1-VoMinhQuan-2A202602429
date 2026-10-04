"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES, N_CLASSES = 54, 7
N_TRAIN_FULL, N_EVAL = 464_809, 116_203


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz. Trả về X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz", allow_pickle=True)
    ev = np.load(f"{processed_dir}/eval.npz", allow_pickle=True)
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    assert X_train.shape == (N_TRAIN_FULL, N_FEATURES) and X_train.dtype == np.float32
    assert X_eval.shape == (N_EVAL, N_FEATURES) and X_eval.dtype == np.float32
    assert y_train.dtype == np.int64 and y_eval.dtype == np.int64
    assert y_train.min() == 0 and y_train.max() == N_CLASSES - 1
    assert len(eval_row_id) == N_EVAL and len(np.unique(eval_row_id)) == N_EVAL
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về X_tr, y_tr, X_val, y_val."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean/std của 10 cột số, tính CHỈ trên phần train còn lại.

    Tính trên val/eval sẽ làm rò rỉ thông tin của tập dùng để đánh giá vào bước tiền xử lý,
    khiến điểm val/eval lạc quan hơn thực tế.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean, std = num.mean(axis=0), num.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # tránh chia 0 nếu một cột là hằng số
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X: 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    Xs = X.copy()
    Xs[:, :N_NUMERIC] = (Xs[:, :N_NUMERIC] - mean) / std
    return Xs


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """load -> tách val -> chuẩn hoá (thống kê của X_tr) -> đưa TOÀN BỘ lên device một lần."""
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)  # chỉ X_tr, không val, không eval
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    def t(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=t(X_tr, torch.float32), y_tr=t(y_tr, torch.int64),
        X_val=t(X_val, torch.float32), y_val=t(y_val, torch.int64),
        X_eval=t(X_eval, torch.float32), y_eval=t(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )
    if verbose:
        maj = np.bincount(y_tr).argmax()
        print(f"X_tr {tuple(X_tr.shape)}  X_val {tuple(X_val.shape)}  X_eval {tuple(X_eval.shape)}")
        for name, yy in (("train", y_tr), ("val", y_val), ("eval", y_eval)):
            frac = np.bincount(yy, minlength=N_CLASSES) / len(yy)
            print(f"  tỉ lệ lớp {name:5s}: " + " ".join(f"{p:.4f}" for p in frac))
        print(f"Đoán luôn lớp đa số ({maj}) trên val: accuracy = {(y_val == maj).mean():.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối nhỏ hơn batch_size vẫn được giữ (không bỏ mẫu nào); với N = 371 847 và batch 512
    lô cuối có 135 mẫu, ảnh hưởng không đáng kể.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device="cpu").to(X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

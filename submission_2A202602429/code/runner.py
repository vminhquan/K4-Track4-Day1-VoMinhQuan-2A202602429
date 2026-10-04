"""runner.py — tiện ích cho notebook: chạy một cfg qua run_experiment, lưu JSON + ảnh + checkpoint,
và dùng lại kết quả đã có nếu cấu hình huấn luyện y hệt (Colab ngắt kết nối thì chạy lại notebook
sẽ không phải huấn luyện lại những thí nghiệm đã xong).

Checkpoint (best_state) lưu ở <REPO_ROOT>/checkpoints/, NGOÀI thư mục nộp (không nộp file .pt).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch

from plots import plot_run
from results_table import load_result, save_result
from train import DEFAULT_CFG, run_experiment

LABEL_KEYS = ("exp_id", "group", "description")  # không ảnh hưởng huấn luyện


def train_key(cfg: dict) -> str:
    c = {k: v for k, v in {**DEFAULT_CFG, **cfg}.items() if k not in LABEL_KEYS}
    c["hidden"] = list(c["hidden"])
    return json.dumps(c, sort_keys=True)


class Runner:
    def __init__(self, data: dict, out_dir: str, ckpt_dir: str, force: bool = False, verbose: bool = False):
        self.data, self.force, self.verbose = data, force, verbose
        self.results_dir, self.fig_dir = Path(out_dir) / "results", Path(out_dir) / "figures"
        self.ckpt_dir = Path(ckpt_dir)
        for d in (self.results_dir, self.fig_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict] = {}

    # ---------- cache ----------
    def _load(self, exp_id: str):
        p = self.results_dir / f"{exp_id}.json"
        if not p.exists():
            return None
        r = load_result(str(p))
        ck = self.ckpt_dir / f"{exp_id}.pt"
        if ck.exists():
            r["best_state"] = torch.load(ck, map_location=self.data["X_tr"].device)
        elif r["summary"]["best_epoch"] > 0:
            return None  # thiếu checkpoint -> chạy lại
        else:
            r["best_state"] = None
        return r

    def _find_same(self, cfg: dict, require_device: str | None = None):
        # require_device: chỉ dùng lại kết quả chạy trên đúng loại thiết bị (vd. "cuda" cho thí nghiệm AMP,
        # vì thời gian/bộ nhớ đo trên CPU không so được với GPU)
        key = train_key(cfg)
        ok = lambda r: require_device is None or r["summary"].get("device") == require_device
        for r in self.results.values():
            if train_key(r["cfg"]) == key and ok(r):
                return r
        own = self.results_dir / f"{cfg['exp_id']}.json"  # ưu tiên file của chính exp_id này
        for p in [own] * own.exists() + sorted(self.results_dir.glob("*.json")):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if train_key(r["cfg"]) == key and ok(r):
                return self._load(p.stem)
        return None

    # ---------- API ----------
    def run(self, cfg: dict, notes: str = "", require_device: str | None = None) -> dict:
        cfg = {**DEFAULT_CFG, **cfg}
        cfg["hidden"] = tuple(cfg["hidden"])
        exp_id = cfg["exp_id"]
        res = None if self.force else self._find_same(cfg, require_device)
        reused = res is not None
        if res is None:
            res = run_experiment(cfg, self.data, verbose=self.verbose)
        else:  # cùng cấu hình huấn luyện: chỉ đổi nhãn
            res = {**res, "cfg": {**res["cfg"], **{k: cfg[k] for k in LABEL_KEYS}}}
            res["cfg"]["hidden"] = tuple(res["cfg"]["hidden"])
        res["notes"] = notes
        save_result(res, str(self.results_dir))
        if res["best_state"] is not None:
            torch.save(res["best_state"], self.ckpt_dir / f"{exp_id}.pt")
        plot_run(res, str(self.fig_dir / f"{exp_id}.png"))
        self.results[exp_id] = res
        s = res["summary"]
        print(f"{'(cache) ' if reused else ''}{exp_id:28s} best_ep {s['best_epoch']:2d}  "
              f"val_loss {s['best_val_loss']:.4f}  val_acc {s['val_acc']:.4f}  "
              f"val_f1 {s['val_macro_f1']:.4f}  {s['time_per_epoch_s']:.2f}s/ep"
              f"{'  DIVERGED' if s['diverged'] else ''}")
        return res

    def drop(self, exp_id: str) -> None:
        """Bỏ một exp_id khỏi bảng (xoá JSON, ảnh, checkpoint)."""
        self.results.pop(exp_id, None)
        for p in (self.results_dir / f"{exp_id}.json", self.fig_dir / f"{exp_id}.png",
                  self.ckpt_dir / f"{exp_id}.pt"):
            if p.exists():
                os.remove(p)

    def table(self, ids=None) -> pd.DataFrame:
        ids = ids or list(self.results)
        rows = []
        for i in ids:
            r = self.results[i]
            c, s = r["cfg"], r["summary"]
            rows.append(dict(exp_id=i, opt=c["optimizer"], lr=c["lr"], batch=c["batch"],
                             hidden="-".join(map(str, c["hidden"])), drop=c["dropout"], clip=c["clip_norm"],
                             prec=c["precision"], init=c["init"], seed=c["seed"],
                             step0=s["step0_loss"], best_ep=s["best_epoch"], best_val_loss=s["best_val_loss"],
                             train_loss=s["final_train_loss"], val_loss=s["final_val_loss"],
                             val_acc=s["val_acc"], val_f1=s["val_macro_f1"],
                             gn_mean=sum(r["history"]["grad_norm"]) / max(1, len(r["history"]["grad_norm"])),
                             s_per_ep=s["time_per_epoch_s"], mem_MB=s["peak_mem_MB"], diverged=s["diverged"]))
        return pd.DataFrame(rows).set_index("exp_id").round(4)

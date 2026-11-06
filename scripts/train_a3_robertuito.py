"""Fine-tuning A3: RoBERTuito sobre la partición por artículo.

Uso:
  python scripts/train_a3_robertuito.py              # entrenamiento completo
  python scripts/train_a3_robertuito.py --fast        # humo: pocos ejemplos / 1 época
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
MODEL_ID = "pysentimiento/robertuito-base-uncased"
MAX_LEN = 128
SEED = 42


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def metrics_at(y_true, y_prob, thr=0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "thr": float(thr),
        "f1_hateful": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "report": classification_report(y_true, y_pred, digits=3),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def best_threshold(y_true, y_prob) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def load_splits(fast: bool):
    sp = {n: pd.read_parquet(ART / f"split_articulo_{n}.parquet") for n in ["train", "dev", "test"]}
    if fast:
        sp["train"] = sp["train"].sample(n=min(4000, len(sp["train"])), random_state=SEED)
        sp["dev"] = sp["dev"].sample(n=min(1000, len(sp["dev"])), random_state=SEED)
        # test se deja completo solo al final del modo full; en fast también lo reducimos
        sp["test"] = sp["test"].sample(n=min(1500, len(sp["test"])), random_state=SEED)
    for n, d in sp.items():
        print(n, len(d), "pct hateful", round(d.HATEFUL.mean() * 100, 1))
    return sp


def to_hf(df: pd.DataFrame, tok) -> Dataset:
    texts = [preprocess_tweet(t, lang="es") for t in df.text.tolist()]
    enc = tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    enc["labels"] = df.HATEFUL.astype(int).tolist()
    return Dataset.from_dict(enc)


def predict_proba(trainer: Trainer, ds: Dataset) -> np.ndarray:
    pred = trainer.predict(ds)
    logits = pred.predictions
    # binary: logits shape (n, 2)
    if logits.ndim == 2 and logits.shape[1] == 2:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e[:, 1] / e.sum(axis=1)).astype(np.float64)
    return (1 / (1 + np.exp(-logits.ravel()))).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="subset + 1 época para validar el pipeline")
    ap.add_argument("--epochs", type=float, default=None)
    args = ap.parse_args()
    set_seed(SEED)
    device = pick_device()
    print("device:", device, "| fast:", args.fast)

    sp = load_splits(args.fast)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    ds_tr = to_hf(sp["train"], tok)
    ds_dv = to_hf(sp["dev"], tok)
    ds_te = to_hf(sp["test"], tok)

    # pesos de clase aproximados vía sample weights no van directo en Trainer simple;
    # usamos class imbalance con label smoothing off y umbral en DEV.
    epochs = args.epochs if args.epochs is not None else (1.0 if args.fast else 3.0)
    bs = 8 if device == "cpu" else 16
    accum = 4 if device == "cpu" else 1
    steps_per_epoch = max(1, (len(ds_tr) // (bs * accum)))
    warmup_steps = max(1, int(0.06 * steps_per_epoch * epochs))
    out_dir = ART / ("a3_fast_run" if args.fast else "a3_run")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        gradient_accumulation_steps=accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        seed=SEED,
        report_to=[],
        fp16=device == "cuda",
        use_cpu=device == "cpu",
        dataloader_num_workers=0,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if logits.ndim == 2:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob = e[:, 1] / e.sum(axis=1)
        else:
            prob = 1 / (1 + np.exp(-logits))
        pred = (prob >= 0.5).astype(int)
        return {
            "f1_hateful": f1_score(labels, pred, pos_label=1),
            "roc_auc": roc_auc_score(labels, prob),
        }

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_dv,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    t0 = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t0
    print("train seconds", round(train_seconds, 1), train_out.metrics)

    # curvas si existen logs
    hist = trainer.state.log_history
    losses = [x["loss"] for x in hist if "loss" in x and "eval_loss" not in x]
    eval_losses = [x["eval_loss"] for x in hist if "eval_loss" in x]
    if losses or eval_losses:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if losses:
            ax.plot(losses, label="train_loss_steps")
        if eval_losses:
            ax.plot(eval_losses, label="eval_loss_epoch")
        ax.legend()
        ax.set_title("A3 RoBERTuito")
        fig.tight_layout()
        fig.savefig(ART / ("a3_fast_loss_curve.png" if args.fast else "a3_loss_curve.png"), dpi=120)
        plt.close(fig)

    print("=== DEV threshold ===")
    prob_dv = predict_proba(trainer, ds_dv)
    y_dv = np.array(ds_dv["labels"])
    thr = best_threshold(y_dv, prob_dv)
    m_dv = metrics_at(y_dv, prob_dv, thr)
    print("thr", thr, "f1", m_dv["f1_hateful"], "auc", m_dv["roc_auc"])

    print("=== TEST ===")
    prob_te = predict_proba(trainer, ds_te)
    y_te = np.array(ds_te["labels"])
    m_te = metrics_at(y_te, prob_te, thr)
    print(m_te["report"])
    print("cm", m_te["cm"], "roc_auc", m_te["roc_auc"], "f1", m_te["f1_hateful"])

    # latencia
    model.eval()
    sample = {k: torch.tensor(v[:1]) for k, v in ds_te.to_dict().items() if k != "labels"}
    # move
    if device != "cpu":
        model.to(device)
        sample = {k: v.to(device) for k, v in sample.items()}
    with torch.no_grad():
        t_lat = time.time()
        for i in range(20):
            _ = model(**{k: torch.tensor(ds_te[k][i : i + 1]).to(model.device) for k in ("input_ids", "attention_mask")})
        lat_ms = (time.time() - t_lat) / 20 * 1000

    save_dir = ART / ("a3_fast_model" if args.fast else "a3_model")
    trainer.save_model(str(save_dir))
    tok.save_pretrained(str(save_dir))

    out = {
        "model": "A3_RoBERTuito",
        "base": MODEL_ID,
        "seed": SEED,
        "fast": args.fast,
        "device": device,
        "epochs": epochs,
        "max_len": MAX_LEN,
        "n_train": len(sp["train"]),
        "n_dev": len(sp["dev"]),
        "n_test": len(sp["test"]),
        "train_seconds": round(train_seconds, 1),
        "latency_ms_per_example_est": round(lat_ms, 2),
        "dev_metrics": {k: m_dv[k] for k in ("f1_hateful", "f1_macro", "roc_auc", "cm", "thr")},
        "test_metrics": {
            "f1_hateful": m_te["f1_hateful"],
            "f1_macro": m_te["f1_macro"],
            "roc_auc": m_te["roc_auc"],
            "cm": m_te["cm"],
            "thr": thr,
            "report": m_te["report"],
        },
        "note": "fast=True usa subset; métricas no comparables con A1/A2 hasta corrida full (Colab GPU recomendado).",
    }
    name = "a3_fast_metrics.json" if args.fast else "a3_metrics.json"
    (ART / name).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado", ART / name, save_dir)
    print("A3 OK")


if __name__ == "__main__":
    main()

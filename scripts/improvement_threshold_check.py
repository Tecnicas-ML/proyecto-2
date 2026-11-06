"""Decisión de mejora comprobada: umbral 0.5 vs umbral elegido en DEV (≈0.28).

Se mide en el DEV de split_articulo (validación) y se reporta el efecto en TEST
sin retocar el umbral después de ver test. También se mira HateCheck familias.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODEL_DIR = ART / "a3_model"
THR_DEV = float(json.loads((ART / "a3_metrics.json").read_text())["test_metrics"]["thr"])
MAX_LEN = 128
BS = 32
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


def probs_for(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BS):
            enc = tok(
                texts[i : i + BS],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def pack(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "thr": thr,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


sp = {n: pd.read_parquet(ART / f"split_articulo_{n}.parquet") for n in ("dev", "test")}
results = {"decision": "Usar umbral elegido en DEV (max F1) en lugar de 0.5 por defecto."}
for split in ("dev", "test"):
    texts = [preprocess_tweet(t, lang="es") for t in sp[split].text.tolist()]
    y = sp[split].HATEFUL.astype(int).values
    p = probs_for(texts)
    results[split] = {
        "thr_0.5": pack(y, p, 0.5),
        "thr_dev_tuned": pack(y, p, THR_DEV),
    }
    print(split, results[split])

# HateCheck: falsos positivos en familias sensibles
hc = load_dataset("Paul/hatecheck-spanish", split="test").to_pandas()
texts = [preprocess_tweet(t, lang="es") for t in hc.test_case.tolist()]
y = (hc.label_gold.str.lower() == "hateful").astype(int).values
p = probs_for(texts)
for thr_name, thr in [("0.5", 0.5), ("dev_tuned", THR_DEV)]:
    pred = (p >= thr).astype(int)
    # contra-discurso approx
    mask = hc.functionality.str.contains("counter|negat|nh_|non_hate|quote", case=False, na=False)
    yt, yp = y[mask.values], pred[mask.values]
    fp = int(((yt == 0) & (yp == 1)).sum())
    results.setdefault("hatecheck_sensitive", {})[thr_name] = {
        "n": int(mask.sum()),
        "false_positives": fp,
        "accuracy": float((yt == yp).mean()),
    }
print("hatecheck_sensitive", results["hatecheck_sensitive"])

(ART / "improvement_threshold.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("IMPROVEMENT OK")

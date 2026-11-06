"""Línea base L sin ajuste: pysentimiento/robertuito-hate-speech sobre el mismo test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODEL_ID = "pysentimiento/robertuito-hate-speech"
SEED = 42
MAX_LEN = 128


def pick_device() -> str:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1  # CPU for pipeline


def best_threshold(y_true, y_prob) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def metrics_at(y_true, y_prob, thr) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "thr": float(thr),
        "f1_hateful": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "report": classification_report(y_true, y_pred, digits=3),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def hate_prob_from_output(out_list) -> float:
    """Soporta id2label binario o multietiqueta del modelo hate-speech."""
    # pipeline text-classification con top_k=None devuelve lista de dicts
    scores = {d["label"].lower(): float(d["score"]) for d in out_list}
    # casos comunes
    for key in ("hateful", "hate", "offensive", "LABEL_1", "label_1"):
        if key.lower() in scores:
            return scores[key.lower()]
    # si hay varias etiquetas de odio, tomar max entre ellas excepto 'others'/'not_*'
    hate_keys = [k for k in scores if not k.startswith("not") and k not in ("others", "ok", "normal")]
    if hate_keys:
        return max(scores[k] for k in hate_keys)
    # fallback: score de la clase predicha si solo hay 2
    if len(scores) == 2:
        return max(scores.values())
    return max(scores.values())


def main():
    device = pick_device()
    print("pipeline device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    print("id2label:", model.config.id2label)
    print("problem_type:", getattr(model.config, "problem_type", None))
    print("num_labels:", model.config.num_labels)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=device,
        truncation=True,
        max_length=MAX_LEN,
        top_k=None,
    )

    sp = {n: pd.read_parquet(ART / f"split_articulo_{n}.parquet") for n in ["train", "dev", "test"]}
    # elegir umbral en DEV (misma partición que A1/A2/A3), test una vez
    for split_name in ("dev", "test"):
        df = sp[split_name]
        texts = [preprocess_tweet(t, lang="es") for t in df.text.tolist()]
        y = df.HATEFUL.astype(int).values
        probs = []
        t0 = time.time()
        # por lotes
        bs = 32
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            outs = clf(batch)
            # outs: list[list[dict]] si top_k=None
            for o in outs:
                probs.append(hate_prob_from_output(o))
            if (i // bs) % 20 == 0:
                print(split_name, i, "/", len(texts))
        probs = np.array(probs, dtype=np.float64)
        elapsed = time.time() - t0
        if split_name == "dev":
            thr = best_threshold(y, probs)
            m_dv = metrics_at(y, probs, thr)
            print("DEV thr", thr, "f1", m_dv["f1_hateful"], "auc", m_dv["roc_auc"])
            lat_ms = elapsed / len(texts) * 1000
        else:
            m_te = metrics_at(y, probs, thr)
            print(m_te["report"])
            print("TEST cm", m_te["cm"], "auc", m_te["roc_auc"], "f1", m_te["f1_hateful"])

    out = {
        "model": "L_robertuito_hate_speech",
        "base": MODEL_ID,
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "seed": SEED,
        "device": str(device),
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
        "note": "Sin ajuste de pesos. Umbral elegido en DEV de split_articulo. Definición de etiquetas del modelo puede diferir de HATEFUL.",
    }
    (ART / "baseline_l_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado artifacts/baseline_l_metrics.json")
    print("BASELINE L OK")


if __name__ == "__main__":
    main()

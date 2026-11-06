"""Entrena A2: BiLSTM sobre secuencias con vocabulario propio (partición por artículo)."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


def limpiar(t: str, minus=True, quitar_tildes=False, separar_puntuacion=True) -> str:
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"@\w+", " @usuario ", t)
    if minus:
        t = t.lower()
    if quitar_tildes:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    if separar_puntuacion:
        t = re.sub(r"([!?¡¿.,;:()\"])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def metrics(y_true, y_prob, thr=0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "thr": thr,
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


print("=== Cargar splits ===")
sp = {n: pd.read_parquet(ART / f"split_articulo_{n}.parquet") for n in ["train", "dev", "test"]}
limpio = {n: sp[n].text.map(limpiar).values for n in sp}
y = {n: sp[n].HATEFUL.values.astype(np.float32) for n in sp}
pesos = compute_class_weight("balanced", classes=np.array([0, 1]), y=y["train"].astype(int))
class_weight = {0: float(pesos[0]), 1: float(pesos[1])}
print("class_weight", class_weight)

n_tok = np.array([len(t.split()) for t in limpio["train"]])
SEQ_LEN = int(np.percentile(n_tok, 95))
MAX_TOKENS = 30_000
print("SEQ_LEN", SEQ_LEN)

vec = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN, standardize=None, split="whitespace"
)
vec.adapt(limpio["train"])
X = {n: vec(limpio[n]).numpy() for n in sp}
vocab_size = len(vec.get_vocabulary())
print("vocab", vocab_size, {n: a.shape for n, a in X.items()})


def build_model(emb_dim=64, lstm_units=64, dropout=0.3, lr=1e-3):
    inp = tf.keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


grid = [
    {"emb_dim": 64, "lstm_units": 64, "dropout": 0.3, "lr": 1e-3},
    {"emb_dim": 100, "lstm_units": 64, "dropout": 0.4, "lr": 1e-3},
    {"emb_dim": 64, "lstm_units": 128, "dropout": 0.3, "lr": 5e-4},
]

rows = []
best = None
t0 = time.time()
for params in grid:
    print("\n=== config", params, "===")
    model = build_model(**params)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
    ]
    hist = model.fit(
        X["train"],
        y["train"],
        validation_data=(X["dev"], y["dev"]),
        epochs=8,
        batch_size=128,
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )
    prob = model.predict(X["dev"], batch_size=256, verbose=0).ravel()
    thr = best_threshold(y["dev"], prob)
    m = metrics(y["dev"], prob, thr)
    row = {**params, "thr": thr, "f1_hateful": m["f1_hateful"], "f1_macro": m["f1_macro"], "roc_auc": m["roc_auc"], "epochs_run": len(hist.history["loss"])}
    rows.append(row)
    print(row)
    if best is None or row["f1_hateful"] > best["row"]["f1_hateful"]:
        best = {"row": row, "model": model, "hist": hist.history, "thr": thr}

train_seconds = time.time() - t0
print("búsqueda total", round(train_seconds, 1), "s")
print("mejor DEV", best["row"])

# curva
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(best["hist"]["loss"], label="train")
ax[0].plot(best["hist"]["val_loss"], label="dev")
ax[0].set_title("A2 loss")
ax[0].legend()
ax[1].plot(best["hist"]["auc"], label="train")
ax[1].plot(best["hist"]["val_auc"], label="dev")
ax[1].set_title("A2 AUC")
ax[1].legend()
fig.tight_layout()
fig.savefig(ART / "a2_loss_curve.png", dpi=120)
plt.close(fig)

print("\n=== TEST ===")
prob_te = best["model"].predict(X["test"], batch_size=256, verbose=0).ravel()
m_te = metrics(y["test"], prob_te, best["thr"])
print(m_te["report"])
print("cm", m_te["cm"], "roc_auc", m_te["roc_auc"], "f1", m_te["f1_hateful"])

prob_dv = best["model"].predict(X["dev"], batch_size=256, verbose=0).ravel()
m_dv = metrics(y["dev"], prob_dv, best["thr"])

t_lat = time.time()
_ = best["model"].predict(X["test"][:100], batch_size=1, verbose=0)
lat_ms = (time.time() - t_lat) / 100 * 1000

out = {
    "model": "A2_BiLSTM",
    "seed": SEED,
    "seq_len": SEQ_LEN,
    "vocab_size": vocab_size,
    "best_dev": best["row"],
    "grid_results": rows,
    "dev_metrics": {
        "f1_hateful": m_dv["f1_hateful"],
        "f1_macro": m_dv["f1_macro"],
        "roc_auc": m_dv["roc_auc"],
        "cm": m_dv["cm"],
        "thr": best["thr"],
    },
    "test_metrics": {
        "f1_hateful": m_te["f1_hateful"],
        "f1_macro": m_te["f1_macro"],
        "roc_auc": m_te["roc_auc"],
        "cm": m_te["cm"],
        "thr": best["thr"],
        "report": m_te["report"],
    },
    "train_search_seconds": round(train_seconds, 1),
    "latency_ms_per_example_est": round(lat_ms, 2),
}
(ART / "a2_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(rows).to_csv(ART / "a2_grid_results.csv", index=False)
best["model"].save(ART / "a2_model.keras")
# guardar vocabulary para inferencia
(ART / "a2_vocab.json").write_text(json.dumps(vec.get_vocabulary(), ensure_ascii=False), encoding="utf-8")
print("guardado a2_metrics.json a2_model.keras")
print("A2 OK")

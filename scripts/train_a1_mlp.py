"""Entrena A1: MLP denso sobre TF-IDF (partición por artículo)."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
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
    # precision_recall_curve returns thresholds of len = len(p)-1
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


print("=== Cargar splits ===")
sp = {n: pd.read_parquet(ART / f"split_articulo_{n}.parquet") for n in ["train", "dev", "test"]}
for n, d in sp.items():
    print(n, d.shape, "pct hateful", round(d.HATEFUL.mean() * 100, 1))

# Stopwords sin negaciones (experimento)
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

negaciones = {"no", "ni", "nunca", "jamás", "sin", "nada", "nadie", "tampoco"}
sw_sin_neg = sorted(set(stopwords.words("spanish")) - negaciones)

print("\n=== TF-IDF + búsqueda acotada en DEV ===")
grid = {
    "stop_words": [None, sw_sin_neg],
    "hidden": [(256,), (128, 64)],
    "alpha": [1e-4, 1e-3],
}
rows = []
best = None
t0 = time.time()
for params in ParameterGrid(grid):
    stop = params["stop_words"]
    tfidf = TfidfVectorizer(
        preprocessor=limpiar,
        token_pattern=r"(?u)\b\w+\b|[!?¡¿]",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        stop_words=stop,
    )
    X_tr = tfidf.fit_transform(sp["train"].text)
    X_dv = tfidf.transform(sp["dev"].text)
    y_tr = sp["train"].HATEFUL.values
    y_dv = sp["dev"].HATEFUL.values
    sw = compute_sample_weight("balanced", y_tr)

    clf = MLPClassifier(
        hidden_layer_sizes=params["hidden"],
        alpha=params["alpha"],
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=SEED,
        verbose=False,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    prob = clf.predict_proba(X_dv)[:, 1]
    thr = best_threshold(y_dv, prob)
    m = metrics(y_dv, prob, thr)
    row = {
        "stop": "none" if stop is None else "sw_sin_neg",
        "hidden": params["hidden"],
        "alpha": params["alpha"],
        "thr": thr,
        "f1_hateful": m["f1_hateful"],
        "f1_macro": m["f1_macro"],
        "roc_auc": m["roc_auc"],
        "n_iter": int(clf.n_iter_),
        "n_features": int(X_tr.shape[1]),
    }
    rows.append(row)
    print(row)
    if best is None or row["f1_hateful"] > best["row"]["f1_hateful"]:
        best = {"row": row, "clf": clf, "tfidf": tfidf, "thr": thr, "loss_curve": list(clf.loss_curve_)}

train_seconds = time.time() - t0
print(f"búsqueda total: {train_seconds:.1f}s")

# Reentrenar el mejor sobre train completo (ya está entrenado sobre train; guardamos ese)
print("\n=== Mejor en DEV ===")
print(best["row"])

# Curva de pérdida del mejor
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(best["loss_curve"])
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("A1 MLP loss curve (best config)")
fig.tight_layout()
fig.savefig(ART / "a1_loss_curve.png", dpi=120)
plt.close(fig)

# Evaluación final en TEST (una sola vez con el modelo elegido en DEV)
print("\n=== TEST (una vez, modelo elegido en DEV) ===")
X_te = best["tfidf"].transform(sp["test"].text)
y_te = sp["test"].HATEFUL.values
prob_te = best["clf"].predict_proba(X_te)[:, 1]
m_te = metrics(y_te, prob_te, best["thr"])
print(m_te["report"])
print("cm", m_te["cm"], "roc_auc", m_te["roc_auc"], "f1_hateful", m_te["f1_hateful"])

# También métricas DEV del mejor para la tabla
X_dv = best["tfidf"].transform(sp["dev"].text)
y_dv = sp["dev"].HATEFUL.values
prob_dv = best["clf"].predict_proba(X_dv)[:, 1]
m_dv = metrics(y_dv, prob_dv, best["thr"])

# Latencia aproximada (CPU, lote 1)
t_lat = time.time()
_ = best["clf"].predict_proba(best["tfidf"].transform(sp["test"].text.iloc[:100]))
lat_ms = (time.time() - t_lat) / 100 * 1000

out = {
    "model": "A1_MLP_TFIDF",
    "seed": SEED,
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
(ART / "a1_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(rows).to_csv(ART / "a1_grid_results.csv", index=False)

# Persistir modelo + vectorizer para la app / comparación
import joblib

joblib.dump(
    {"tfidf": best["tfidf"], "clf": best["clf"], "thr": best["thr"], "limpiar": "scripts.train_a1_mlp.limpiar"},
    ART / "a1_model.joblib",
)
print("guardado", ART / "a1_metrics.json", ART / "a1_model.joblib", ART / "a1_loss_curve.png")
print("A1 OK")

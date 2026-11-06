"""Análisis de errores con Paul/hatecheck-spanish usando el modelo A3."""
from __future__ import annotations

import json
from collections import defaultdict
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
THR = float(json.loads((ART / "a3_metrics.json").read_text())["test_metrics"]["thr"])
MAX_LEN = 128
BS = 32

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, "thr", THR)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

ds = load_dataset("Paul/hatecheck-spanish", split="test")
df = ds.to_pandas()
# gold: hateful / non-hateful
y = (df["label_gold"].str.lower() == "hateful").astype(int).values
texts = [preprocess_tweet(t, lang="es") for t in df["test_case"].tolist()]

probs = []
with torch.no_grad():
    for i in range(0, len(texts), BS):
        batch = texts[i : i + BS]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        if i % (BS * 20) == 0:
            print("hatecheck", i, "/", len(texts))
probs = np.concatenate(probs)
pred = (probs >= THR).astype(int)

print(
    "GLOBAL  P=%.3f R=%.3f F1=%.3f"
    % (
        precision_score(y, pred),
        recall_score(y, pred),
        f1_score(y, pred),
    )
)

rows = []
for fn, g in df.groupby("functionality"):
    idx = g.index.to_numpy()
    yt, yp = y[idx], pred[idx]
    # algunas funcionalidades son solo non-hateful o solo hateful
    if len(np.unique(yt)) < 2:
        f1 = float("nan")
        acc = float((yt == yp).mean())
    else:
        f1 = float(f1_score(yt, yp, zero_division=0))
        acc = float((yt == yp).mean())
    rows.append(
        {
            "functionality": fn,
            "n": int(len(g)),
            "pct_hateful_gold": round(float(yt.mean()) * 100, 1),
            "accuracy": round(acc, 3),
            "f1": None if np.isnan(f1) else round(f1, 3),
            "recall_hateful": round(float(recall_score(yt, yp, zero_division=0)), 3) if yt.sum() else None,
            "false_pos": int(((yt == 0) & (yp == 1)).sum()),
            "false_neg": int(((yt == 1) & (yp == 0)).sum()),
        }
    )

res = pd.DataFrame(rows).sort_values(["accuracy", "n"], ascending=[True, False])
res.to_csv(ART / "hatecheck_by_functionality.csv", index=False)

# ejemplos fallidos útiles para el informe
df = df.copy()
df["y"] = y
df["prob"] = probs
df["pred"] = pred
fn_cases = df[df.pred != df.y][["functionality", "test_case", "label_gold", "prob", "target_ident"]]
# priorizar contra-discurso / negaciones si existen
prefer = fn_cases[
    fn_cases.functionality.str.contains("negat|counter|quote|non_hateful|nh_", case=False, na=False)
]
sample_err = pd.concat(
    [
        prefer.head(8),
        fn_cases.head(12),
    ]
).drop_duplicates().head(20)
sample_err.to_csv(ART / "hatecheck_error_examples.csv", index=False)

# resumen por "familia" de funcionalidad
def family(name: str) -> str:
    n = name.lower()
    if "counter" in n or "refute" in n:
        return "contra-discurso"
    if "negat" in n or "nh_" in n or "non_hate" in n:
        return "no-odio / negación"
    if "quote" in n or "mention" in n:
        return "cita / mención"
    if "insult" in n or "derog" in n or "threat" in n or "hate" in n:
        return "odio explícito"
    return "otro"


df["family"] = df["functionality"].map(family)
fam = []
for f, g in df.groupby("family"):
    yt, yp = g["y"].values, g["pred"].values
    fam.append(
        {
            "family": f,
            "n": len(g),
            "accuracy": round(float((yt == yp).mean()), 3),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 3) if len(np.unique(yt)) > 1 else None,
            "false_pos": int(((yt == 0) & (yp == 1)).sum()),
            "false_neg": int(((yt == 1) & (yp == 0)).sum()),
        }
    )
fam_df = pd.DataFrame(fam).sort_values("accuracy")
fam_df.to_csv(ART / "hatecheck_by_family.csv", index=False)

out = {
    "model": "A3",
    "threshold": THR,
    "n": int(len(df)),
    "precision": float(precision_score(y, pred)),
    "recall": float(recall_score(y, pred)),
    "f1": float(f1_score(y, pred)),
    "worst_functionalities": res.head(10).to_dict(orient="records"),
    "families": fam_df.to_dict(orient="records"),
}
(ART / "hatecheck_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nPor familia:")
print(fam_df.to_string(index=False))
print("\nPeores funcionalidades:")
print(res.head(12).to_string(index=False))
print("guardado hatecheck_*.csv/json")
print("HATECHECK OK")

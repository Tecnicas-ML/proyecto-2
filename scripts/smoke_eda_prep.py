"""Smoke test: ejecuta EDA + preparación del notebook fuera de Colab."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from pysentimiento.preprocessing import preprocess_tweet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)


def show(df: pd.DataFrame, n: int = 5) -> None:
    print(df.head(n).to_string(index=False))


print("=== 2. Cargar corpus ===")
ds = load_dataset("piuba-bigdata/contextualized_hate_speech")
print(ds)
print(ds["train"].features)

df = {k: ds[k].to_pandas() for k in ds}
tr, dv, te = df["train"], df["dev"], df["test"]

print("\n=== 3.1 Inventario y calidad ===")
for k, d in df.items():
    vacios = d["text"].isna().sum() + (d["text"].str.strip() == "").sum()
    print(k, d.shape, "| vacíos en text:", vacios, "| duplicados exactos de text:", d["text"].duplicated().sum())

print("\n=== 3.2 Balance de clases ===")
cats = [
    "CALLS",
    "WOMEN",
    "LGBTI",
    "RACISM",
    "CLASS",
    "POLITICS",
    "DISABLED",
    "APPEARANCE",
    "CRIMINAL",
]
for k, d in df.items():
    print(k, d["HATEFUL"].value_counts(normalize=True).round(3).to_dict())
print(tr.loc[tr.HATEFUL == 1, cats].mean().sort_values(ascending=False).round(3))

print("\n=== 3.3 Fuga por artículo ===")
key = "body"
a_tr, a_dv, a_te = set(tr[key]), set(dv[key]), set(te[key])
print("artículos únicos train/dev/test:", len(a_tr), len(a_dv), len(a_te))
print("test con artículo visto en train:", te[key].isin(a_tr).mean().round(3))
print("dev con artículo visto en train:", dv[key].isin(a_tr).mean().round(3))
tasa = tr.groupby(key).HATEFUL.mean()
print("tasa de odio por artículo (train):", tasa.describe().round(3).to_dict())

print("\n=== Partición por artículo ===")
full = pd.concat(
    [
        tr.assign(split_oficial="train"),
        dv.assign(split_oficial="dev"),
        te.assign(split_oficial="test"),
    ],
    ignore_index=True,
)
groups = full[key].factorize()[0]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_trdv, idx_te = next(gss.split(full, groups=groups))
rest = full.iloc[idx_trdv]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
i_tr, i_dv = next(gss2.split(rest, groups=groups[idx_trdv]))
g_tr, g_dv, g_te = rest.iloc[i_tr], rest.iloc[i_dv], full.iloc[idx_te]
for n, d in [("train", g_tr), ("dev", g_dv), ("test", g_te)]:
    print(n, len(d), "| artículos:", d[key].nunique(), "| % hateful:", round(d.HATEFUL.mean() * 100, 1))
assert not (set(g_tr[key]) & set(g_te[key])) and not (set(g_tr[key]) & set(g_dv[key]))

cols = ["id", "text", "title", "context_tweet", "HATEFUL"] + cats
for n, d in [("train", g_tr), ("dev", g_dv), ("test", g_te)]:
    path = OUT / f"split_articulo_{n}.parquet"
    d[cols].to_parquet(path, index=False)
    print("guardado", path)

print("\n=== 3.4 Longitud (muestra 2000) ===")
tr = tr.copy()
tr["n_words"] = tr.text.str.split().str.len()
print(tr.n_words.describe().round(1))
print(tr.groupby("HATEFUL").n_words.mean().round(2))
tok = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")
muestra = [preprocess_tweet(t, lang="es") for t in tr.text.sample(2000, random_state=0)]
lens = [len(x) for x in tok(muestra)["input_ids"]]
print("tokens p50/p95/max:", np.percentile(lens, [50, 95, 100]), "| límite del modelo: 128")

print("\n=== 3.5 Muestras ===")
print("HATEFUL=1")
show(tr[tr.HATEFUL == 1].sample(5, random_state=1)[["text"]])
print("HATEFUL=0")
show(tr[tr.HATEFUL == 0].sample(5, random_state=1)[["text"]])

print("\n=== 3.6 Regla de groserías en DEV oficial ===")
groserias = [
    "mierda",
    "puto",
    "puta",
    "boludo",
    "pelotudo",
    "idiota",
    "imbecil",
    "imbécil",
    "estupido",
    "estúpido",
    "basura",
    "forro",
    "negro de mierda",
    "kuka",
    "zurdo",
    "gorila",
    "chorro",
    "vago",
    "asco",
    "mamerto",
]
patron = re.compile(r"\b(" + "|".join(map(re.escape, groserias)) + r")\b", re.IGNORECASE)
pred = dv.text.str.contains(patron).astype(int)
print(
    "regla  P=%.3f R=%.3f F1=%.3f"
    % (
        precision_score(dv.HATEFUL, pred),
        recall_score(dv.HATEFUL, pred),
        f1_score(dv.HATEFUL, pred),
    )
)
print(
    "odio SIN palabra de la lista:",
    ((dv.HATEFUL == 1) & (pred == 0)).sum(),
    "| palabra de la lista SIN odio:",
    ((dv.HATEFUL == 0) & (pred == 1)).sum(),
)

print("\n=== 4.1 Cargar partición fija ===")
sp = {n: pd.read_parquet(OUT / f"split_articulo_{n}.parquet") for n in ["train", "dev", "test"]}
for n, d in sp.items():
    print(n, d.shape, "| % hateful:", round(d.HATEFUL.mean() * 100, 1))

print("\n=== 4.2 Mayúsculas / emojis ===")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def prop_mayus(t: str) -> float:
    w = re.findall(r"[^\W\d_]{3,}", t)
    return float(np.mean([x.isupper() for x in w])) if w else 0.0


d = sp["train"]
print("prop. palabras en MAYÚSCULAS por clase:", d.text.map(prop_mayus).groupby(d.HATEFUL).mean().round(3).to_dict())
print("% con emoji por clase:", d.text.str.contains(EMOJI).groupby(d.HATEFUL).mean().round(3).to_dict())
print(
    "% con URL:",
    round(d.text.str.contains(r"https?://").mean() * 100, 1),
    "| % con @usuario:",
    round(d.text.str.contains("@usuario").mean() * 100, 1),
)


def limpiar(t, minus=True, quitar_tildes=False, separar_puntuacion=True):
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"@\w+", " @usuario ", t)
    if minus:
        t = t.lower()
    if quitar_tildes:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    if separar_puntuacion:
        t = re.sub(r"([!?¡¿.,;:()\"])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def prep_a3(t):
    return preprocess_tweet(t, lang="es")


print("\n=== 4.3 Antes/después (muestra) ===")
casos = []
for subset in [
    d[d.text.str.contains(r"\b(no|nunca|ni)\b", case=False)],
    d[d.text.map(prop_mayus) > 0.5],
    d[d.text.str.contains(EMOJI)],
    d[d.text.str.contains(r"https?://")],
]:
    if len(subset):
        casos.append(subset.sample(min(2, len(subset)), random_state=0))
casos = pd.concat(casos)
show(
    pd.DataFrame(
        {
            "original": casos.text,
            "limpiar": casos.text.map(limpiar),
            "prep_a3": casos.text.map(prep_a3),
            "HATEFUL": casos.HATEFUL,
        }
    ),
    n=8,
)

print("\n=== 4.4 Stopwords / negaciones ===")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

sw = set(stopwords.words("spanish"))
negaciones = ["no", "ni", "nunca", "jamás", "sin", "nada", "nadie", "tampoco"]
print("negaciones dentro de la lista NLTK:", [w for w in negaciones if w in sw])
sw_sin_neg = sorted(sw - set(negaciones))
print("stopwords sin negaciones:", len(sw_sin_neg))

print("\n=== 4.5 Tokenización A3 ===")
tok3 = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")
print("¿tokenizador lowercasing?", tok3.tokenize("HOLA MUNDO") == tok3.tokenize("hola mundo"))
MAX_LEN = 128
# medir truncamiento en una muestra para no saturar RAM/tiempo
sample_n = min(5000, len(sp["train"]))
sample = sp["train"].sample(sample_n, random_state=0)
txt = [prep_a3(t) for t in sample.text]
ctx = [prep_a3(t) for t in sample.title]
l_txt = np.array([len(x) for x in tok3(txt, truncation=False)["input_ids"]])
l_par = np.array([len(x) for x in tok3(txt, ctx, truncation=False)["input_ids"]])
print("solo comentario  p95=%d  truncados=%.2f%% (n=%d)" % (np.percentile(l_txt, 95), (l_txt > MAX_LEN).mean() * 100, sample_n))
print(
    "comentario+título p95=%d truncados=%.2f%% (n=%d)"
    % (np.percentile(l_par, 95), (l_par > MAX_LEN).mean() * 100, sample_n)
)

print("\n=== 4.6 TF-IDF A1 ===")


def hacer_tfidf(stop=None):
    return TfidfVectorizer(
        preprocessor=limpiar,
        token_pattern=r"(?u)\b\w+\b|[!?¡¿]",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        stop_words=stop,
    )


tfidf = hacer_tfidf()
X1 = {"train": tfidf.fit_transform(sp["train"].text)}
X1.update({n: tfidf.transform(sp[n].text) for n in ["dev", "test"]})
print({n: X.shape for n, X in X1.items()})

print("\n=== 4.7 Vocabulario A2 ===")
limpio = {n: sp[n].text.map(limpiar).values for n in sp}
n_tok = np.array([len(t.split()) for t in limpio["train"]])
SEQ_LEN = int(np.percentile(n_tok, 95))
vec = tf.keras.layers.TextVectorization(
    max_tokens=30_000, output_sequence_length=SEQ_LEN, standardize=None, split="whitespace"
)
vec.adapt(limpio["train"])
X2 = {n: vec(limpio[n]).numpy() for n in sp}
vocab = set(vec.get_vocabulary())
oov = np.mean([w not in vocab for t in limpio["dev"] for w in t.split()])
print(
    "SEQ_LEN:",
    SEQ_LEN,
    "| vocabulario:",
    len(vocab),
    "| pct palabras de dev fuera del vocabulario: %.1f" % (oov * 100),
)
print("pct comentarios train truncados:", round((n_tok > SEQ_LEN).mean() * 100, 1))
print({n: arr.shape for n, arr in X2.items()})

print("\n=== 4.8 Class weights + prep_config ===")
y = {n: sp[n].HATEFUL.values for n in sp}
pesos = compute_class_weight("balanced", classes=np.array([0, 1]), y=y["train"])
print("pesos de clase (train):", pesos.round(3))
prep_config = {
    "split": "split_articulo",
    "a3_model": "pysentimiento/robertuito-base-uncased",
    "a3_max_len": MAX_LEN,
    "a3_preprocess": "pysentimiento.preprocess_tweet(lang='es')",
    "a2_seq_len": SEQ_LEN,
    "limpiar": {"minus": True, "quitar_tildes": False, "separar_puntuacion": True},
    "class_weights": pesos.round(4).tolist(),
    "artifacts_dir": str(OUT),
}
(OUT / "prep_config.json").write_text(json.dumps(prep_config, indent=2, ensure_ascii=False), encoding="utf-8")
print(prep_config)
print("\nSMOKE OK")

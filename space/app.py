"""Hugging Face Space entrypoint (Gradio).

Subir junto con la carpeta `a3_model/` (pesos fine-tuned) al Space.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "a3_model"
META = ROOT / "a3_metrics.json"
MAX_LEN = 128
DEFAULT_THR = 0.28
if META.exists():
    DEFAULT_THR = float(json.loads(META.read_text())["test_metrics"]["thr"])

device = 0 if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
if device != "cpu":
    model.to("cuda")
model.eval()


def predecir(texto: str, umbral: float):
    texto = (texto or "").strip()
    if not texto:
        return "Vacío", 0.0, "Escribí un comentario para analizar."
    prep = preprocess_tweet(texto, lang="es")
    enc = tok(prep, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    if device != "cpu":
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
        prob = torch.softmax(logits, dim=-1)[1].item()
    ocultar = prob >= umbral
    etiqueta = "Con odio (se ocultaría)" if ocultar else "Sin odio (se mostraría)"
    aviso = (
        "Este comentario se ocultaría por las normas de la comunidad. "
        "Queremos un espacio donde todas las personas puedan leer sin tropiezos con agresiones."
        if ocultar
        else "Por debajo del umbral: se publicaría. El modelo no sustituye una decisión humana."
    )
    return etiqueta, round(prob, 4), aviso


demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Textbox(lines=4, label="Comentario en español"),
        gr.Slider(0.05, 0.95, value=DEFAULT_THR, step=0.01, label="Umbral"),
    ],
    outputs=[
        gr.Textbox(label="Decisión"),
        gr.Number(label="Probabilidad de odio"),
        gr.Textbox(label="Mensaje"),
    ],
    title="Menos odio a la vista",
    description="RoBERTuito fine-tuned · español · filtro previo de moderación",
)

if __name__ == "__main__":
    demo.launch()

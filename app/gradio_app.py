"""App Gradio: detecta odio con el modelo A3 fine-tuned.

Local:
  source .venv/bin/activate
  python app/gradio_app.py

Space (HF): subir artifacts/a3_model + este archivo y ajustar MODEL_DIR.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "a3_model"
METRICS = ROOT / "artifacts" / "a3_metrics.json"
MAX_LEN = 128
DEFAULT_THR = 0.28

if METRICS.exists():
    DEFAULT_THR = float(json.loads(METRICS.read_text())["test_metrics"]["thr"])

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


def predecir(texto: str, umbral: float):
    texto = (texto or "").strip()
    if not texto:
        return "Vacío", 0.0, "Escribí un comentario para analizar."
    if len(texto) > 500:
        return "Muy largo", 0.0, "Límite sugerido: 500 caracteres (el modelo admite 128 tokens)."

    prep = preprocess_tweet(texto, lang="es")
    enc = tok(prep, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
        prob = torch.softmax(logits, dim=-1)[1].item()

    ocultar = prob >= umbral
    etiqueta = "Con odio (se ocultaría)" if ocultar else "Sin odio (se mostraría)"
    aviso = (
        "Este comentario se ocultaría por las normas de la comunidad. "
        "Queremos un espacio donde todas las personas —incluidas chicas, niños y adolescentes— "
        "puedan leer sin tropiezos con agresiones. Opción humana: mostrar de todos modos / revisar."
        if ocultar
        else "Por debajo del umbral: se publicaría. El modelo no sustituye una decisión humana."
    )
    return etiqueta, round(prob, 4), aviso


demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Textbox(lines=4, label="Comentario en español", placeholder="Escribí un comentario…"),
        gr.Slider(0.05, 0.95, value=DEFAULT_THR, step=0.01, label="Umbral (elegido en DEV)"),
    ],
    outputs=[
        gr.Textbox(label="Decisión"),
        gr.Number(label="Probabilidad de odio"),
        gr.Textbox(label="Mensaje para la comunidad"),
    ],
    title="Menos odio a la vista",
    description=(
        "Filtro previo con RoBERTuito ajustado al corpus piuba-bigdata. "
        "Idioma: español. No es censura absoluta ni un diagnóstico de salud mental."
    ),
    examples=[
        ["Qué buena nota, gracias por explicar el tema con claridad.", DEFAULT_THR],
        ["Ojalá desaparezcan todos esos negros de mierda de una vez.", DEFAULT_THR],
        ["No estoy de acuerdo con la medida, me parece injusta.", DEFAULT_THR],
    ],
)

if __name__ == "__main__":
    demo.launch()

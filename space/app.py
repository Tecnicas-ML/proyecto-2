"""Hugging Face Space — Menos odio a la vista (modelo A3 RoBERTuito).

ZeroGPU: la inferencia va en una función decorada con @spaces.GPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LOCAL_MODEL = ROOT / "a3_model"
META = ROOT / "a3_metrics.json"
DEFAULT_MODEL_ID = os.environ.get("HF_MODEL_ID", "tatisvivass/menos-odio-a3-robertuito")
MAX_LEN = 128
MAX_CHARS = 500
DEFAULT_THR = 0.28

if META.exists():
    DEFAULT_THR = float(json.loads(META.read_text())["test_metrics"]["thr"])

MODEL_SRC = str(LOCAL_MODEL) if (LOCAL_MODEL / "config.json").exists() else DEFAULT_MODEL_ID
tok = AutoTokenizer.from_pretrained(MODEL_SRC)
# Pesos en CPU al arrancar; ZeroGPU mueve a GPU dentro de @spaces.GPU.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SRC)
model.eval()

LIMITES = (
    f"**Límites:** español · hasta {MAX_CHARS} caracteres (~{MAX_LEN} tokens) · "
    f"umbral por defecto {DEFAULT_THR:.2f} (elegido en DEV) · "
    "no sustituye moderación humana · puede fallar fuera de dominio "
    "(otro país, otro tema, typos, contra-discurso).\n\n"
    f"*Modelo:* `{MODEL_SRC}`"
)

EJEMPLOS = [
    ["Qué buena nota, gracias por explicar el tema con claridad.", DEFAULT_THR],
    ["No estoy de acuerdo con la medida, me parece injusta.", DEFAULT_THR],
    ["Ojalá desaparezcan todos esos negros de mierda de una vez.", DEFAULT_THR],
    ["Hay que echar a todos los inmigrantes del país ya.", DEFAULT_THR],
    ["No odio a nadie: rechazo las ideas, no a las personas.", DEFAULT_THR],
]


@spaces.GPU
def predecir(texto: str, umbral: float):
    """Devuelve (decisión, probabilidad, mensaje). Endpoint API: /predict."""
    texto = (texto or "").strip()
    if not texto:
        return "Vacío", 0.0, "Escribí un comentario para analizar."
    if len(texto) > MAX_CHARS:
        return (
            "Muy largo",
            0.0,
            f"Límite: {MAX_CHARS} caracteres (el modelo admite {MAX_LEN} tokens). "
            "Acortá el texto o se truncará al desplegarse en un blog.",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prep = preprocess_tweet(texto, lang="es")
    enc = tok(prep, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
        prob = float(torch.softmax(logits, dim=-1)[1].item())

    ocultar = prob >= float(umbral)
    if ocultar:
        etiqueta = "Con odio (se ocultaría)"
        aviso = (
            "Este comentario se ocultaría por las normas de la comunidad. "
            "Queremos un espacio donde todas las personas —incluidas chicas, "
            "niños y adolescentes— puedan leer sin tropiezos con agresiones. "
            "Opción humana: mostrar de todos modos / revisar."
        )
    else:
        etiqueta = "Sin odio (se mostraría)"
        aviso = (
            "Por debajo del umbral: se publicaría. "
            "El modelo no sustituye una decisión humana."
        )
    return etiqueta, round(prob, 4), aviso


with gr.Blocks(title="Menos odio a la vista") as demo:
    gr.Markdown(
        "# Menos odio a la vista\n"
        "Filtro previo con **RoBERTuito** ajustado al corpus `piuba-bigdata` "
        "(etiqueta `HATEFUL`). F1 odio en test ≈ **0,653** · AUC ≈ **0,90**.\n\n"
        + LIMITES
    )
    with gr.Row():
        with gr.Column():
            texto = gr.Textbox(
                lines=4,
                label="Comentario en español",
                placeholder="Escribí un comentario…",
                max_lines=8,
            )
            umbral = gr.Slider(
                0.05,
                0.95,
                value=DEFAULT_THR,
                step=0.01,
                label="Umbral (elegido en DEV ≈ 0,28)",
            )
            btn = gr.Button("Analizar", variant="primary")
        with gr.Column():
            decision = gr.Textbox(label="Decisión")
            prob = gr.Number(label="Probabilidad de odio")
            mensaje = gr.Textbox(label="Mensaje para la comunidad", lines=4)

    btn.click(predecir, inputs=[texto, umbral], outputs=[decision, prob, mensaje], api_name="predict")
    texto.submit(predecir, inputs=[texto, umbral], outputs=[decision, prob, mensaje], api_name=False)

    gr.Examples(
        examples=EJEMPLOS,
        inputs=[texto, umbral],
        outputs=[decision, prob, mensaje],
        fn=predecir,
        cache_examples=False,
        label="Ejemplos (limpios, agresivos y difíciles)",
    )
    gr.Markdown(
        "### Cómo se usa en el blog\n"
        "Si la probabilidad ≥ umbral, el comentario se reemplaza por el aviso de "
        "**normas de la comunidad** con botón *mostrar de todos modos*."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()

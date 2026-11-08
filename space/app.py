"""Hugging Face Space — Menos odio a la vista v5 (RoBERTuito multitarea con contexto).

ZeroGPU: la inferencia va en funciones decoradas con @spaces.GPU.
`multitarea_es.py` y `robustez_es.py` son copias exactas de los módulos del cuaderno.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import spaces
import torch
from huggingface_hub import snapshot_download

from multitarea_es import HEADS, Predictor

ROOT = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("HF_MODEL_ID", "tatisvivass/menos-odio-v5-multitarea")
MAX_CHARS = 500

NOMBRES = {"vulgaridad": "Vulgaridad", "insulto": "Insulto / ataque personal", "odio_grupal": "Odio grupal"}


def ruta_modelo() -> str:
    """Carpeta `modelo_v5` local (o MODELO_DIR) si existe; si no, descarga desde el Hub."""
    for candidata in (os.environ.get("MODELO_DIR"), ROOT / "modelo_v5"):
        if candidata and Path(candidata, "multitarea_config.json").exists():
            return str(candidata)
    return snapshot_download(MODEL_ID)


# Pesos en CPU al arrancar; ZeroGPU mueve a GPU dentro de @spaces.GPU.
P = Predictor.cargar(ruta_modelo(), device=torch.device("cpu"))

LIMITES = (
    f"**Límites:** español · hasta {MAX_CHARS} caracteres · umbrales elegidos en DEV_EVAL, no universales · "
    "no sustituye moderación humana · los estereotipos regionales no siempre se reconocen como odio grupal · "
    "vulgaridad e insulto se validaron con muy pocas filas · el efecto del contexto en esas dos señales es "
    "una observación por confirmar.\n\n"
    f"*Modelo:* `{MODEL_ID}`"
)

EJEMPLOS = [
    ["Me encanta la cultura costeña", ""],
    ["Todos los costeños son perezosos", ""],
    ["No está bien decir que todos los costeños son perezosos", ""],
    ["Usted sí es mucho hp, parce", "Conversación entre amigos planeando el asado, todo en buena onda."],
    ["Usted es un hp", "Discusión acalorada: le reclama a otro usuario por una deuda."],
    ["HI J U E P U T A", ""],
    ["hijuep*ta", ""],
    ["Qué parche tan bacano, mijo", ""],
]


def _probs(texto: str, contexto: str | None, tta: bool = True):
    P.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.model.to(P.device)
    return P.probs([texto], [contexto], tta=tta)[0]


def _valida(texto: str) -> str | None:
    if not texto:
        return "Escribí un comentario para analizar."
    if len(texto) > MAX_CHARS:
        return f"Límite: {MAX_CHARS} caracteres. Acortá el texto."
    return None


@spaces.GPU
def analizar(texto, contexto, tta, comparar, *umbrales):
    """Laboratorio: tres señales con umbrales ajustables y efecto del contexto."""
    texto, contexto = (texto or "").strip(), (contexto or "").strip() or None
    error = _valida(texto)
    if error:
        return error, pd.DataFrame()

    con = _probs(texto, contexto, tta)
    sin = _probs(texto, None, tta) if (comparar and contexto) else None
    filas, activas = [], []
    for j, h in enumerate(HEADS):
        p = float(con[j])
        if math.isnan(p):
            filas.append({"categoría": NOMBRES[h], "probabilidad": "sin entrenar", "umbral": "—", "señal": "—"})
            continue
        senal = p >= float(umbrales[j])
        if senal:
            activas.append(NOMBRES[h])
        fila = {"categoría": NOMBRES[h], "probabilidad": round(p, 3), "umbral": round(float(umbrales[j]), 3),
                "señal": "sí" if senal else "no"}
        if sin is not None:
            fila["sin contexto"] = round(float(sin[j]), 3)
            fila["cambio por contexto"] = round(p - float(sin[j]), 3)
        filas.append(fila)
    resumen = f"Señales activas: {', '.join(activas)}" if activas else "Sin señales activas"
    return resumen, pd.DataFrame(filas)


@spaces.GPU
def predecir(texto, contexto):
    """Endpoint API `/predict` que consume el blog. Cada sitio aplica su propia regla sobre las señales."""
    texto, contexto = (texto or "").strip(), (contexto or "").strip() or None
    error = _valida(texto)
    if error:
        return {"estado": "invalido", "mensaje": error, "senales": None}
    probs = _probs(texto, contexto)
    senales = {}
    for j, h in enumerate(HEADS):
        p = float(probs[j])
        senales[h] = {
            "prob": None if math.isnan(p) else round(p, 4),
            "umbral": round(float(P.thr[j]), 4),
            "senal": (not math.isnan(p)) and p >= float(P.thr[j]),
        }
    return {"estado": "ok", "mensaje": "", "senales": senales}


with gr.Blocks(title="Menos odio a la vista · v5") as demo:
    gr.Markdown(
        "# Menos odio a la vista · v5\n"
        "Un único modelo entrega **tres señales independientes** (vulgaridad, insulto, odio grupal) sobre el "
        "comentario y, si existe, su contexto. Son probabilidades calibradas, no un veredicto.\n\n"
        "Odio grupal: F1 ≈ **0,706** en DEV_EVAL, sin diferencia detectable frente al modelo A3 en las mismas "
        "filas (0,689). TEST sin evaluar.\n\n" + LIMITES
    )
    with gr.Row():
        with gr.Column():
            texto = gr.Textbox(lines=4, max_lines=8, label="Comentario en español", placeholder="Escribí un comentario…")
            contexto = gr.Textbox(lines=2, label="Contexto (opcional: título de la noticia o mensaje previo)")
            with gr.Row():
                tta = gr.Checkbox(value=True, label="Usar vista normalizada (TTA)")
                comparar = gr.Checkbox(value=True, label="Comparar con y sin contexto")
            umbrales = [
                gr.Slider(0.05, 0.95, value=round(float(P.thr[j]), 3), step=0.001,
                          label=f"Umbral · {NOMBRES[h]} (DEV_EVAL: {P.thr[j]:.3f})")
                for j, h in enumerate(HEADS)
            ]
            btn = gr.Button("Analizar", variant="primary")
        with gr.Column():
            resumen = gr.Textbox(label="Resultado")
            tabla = gr.Dataframe(label="Señales por categoría", interactive=False)
            gr.Markdown(
                "Las categorías **no son excluyentes**. Mover un umbral cambia el equilibrio entre falsos "
                "positivos y falsos negativos. La decisión final corresponde a una persona."
            )

    entradas = [texto, contexto, tta, comparar, *umbrales]
    btn.click(analizar, inputs=entradas, outputs=[resumen, tabla], api_name=False)
    texto.submit(analizar, inputs=entradas, outputs=[resumen, tabla], api_name=False)

    # Sin interfaz visible: expone POST /gradio_api/call/predict para el blog estático.
    api_salida = gr.JSON(visible=False)
    api_btn = gr.Button(visible=False)
    api_btn.click(predecir, inputs=[texto, contexto], outputs=api_salida, api_name="predict")

    gr.Examples(examples=EJEMPLOS, inputs=[texto, contexto], label="Ejemplos (limpios, agresivos y difíciles)")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()

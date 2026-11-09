"""Carga única del modelo v5: la comparten LA PLAZA y el laboratorio (una sola copia en memoria)."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from huggingface_hub import snapshot_download

from multitarea_es import Predictor

ROOT = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("HF_MODEL_ID", "DavidVargas-Ctrl/menos-odio-v5-multitarea")


def ruta_modelo() -> str:
    """Carpeta `modelo_v5` local (o MODELO_DIR) si existe; si no, descarga desde el Hub."""
    for candidata in (os.environ.get("MODELO_DIR"), ROOT / "modelo_v5"):
        if candidata and Path(candidata, "multitarea_config.json").exists():
            return str(candidata)
    return snapshot_download(MODEL_ID)


@st.cache_resource(show_spinner="Cargando modelo v5 (primera vez puede tardar)…")
def load_predictor() -> Predictor:
    return Predictor.cargar(ruta_modelo(), device=torch.device("cpu"))

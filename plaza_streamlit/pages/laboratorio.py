# -*- coding: utf-8 -*-
"""Laboratorio del modelo v5 (app de la sección 6.3 del cuaderno)."""
import numpy as np
import streamlit as st

import robustez_es as R
from carga_modelo import load_predictor
from multitarea_es import HEADS

st.set_page_config(page_title="Menos odio a la vista · v5", page_icon="🛡️", layout="centered")
st.page_link("streamlit_app.py", label="← Volver a LA PLAZA")

P = load_predictor()

NOMBRES = {"vulgaridad": "Vulgaridad", "insulto": "Insulto / ataque personal", "odio_grupal": "Odio grupal"}
AYUDA = {"vulgaridad": "Lenguaje grosero, aunque no agreda a nadie.",
         "insulto": "Expresión dirigida a agredir a otra persona.",
         "odio_grupal": "Ataque, discriminación o estereotipo despectivo contra un grupo."}
EJEMPLOS = {
    "— elegir ejemplo —": ("", ""),
    "Cultura costeña": ("Me encanta la cultura costeña", ""),
    "Estereotipo": ("Todos los costeños son perezosos", ""),
    "Contradiscurso": ("No está bien decir que todos los costeños son perezosos", ""),
    "hp amistoso": ("Usted sí es mucho hp, parce", "Conversación entre amigos planeando el asado, todo en buena onda."),
    "hp en discusión": ("Usted es un hp", "Discusión acalorada: le reclama a otro usuario por una deuda."),
    "Ofuscado: espaciado": ("HI J U E P U T A", ""), "Ofuscado: leet": ("H1JUEPUTA", ""),
    "Ofuscado: máscara": ("hijuep*ta", ""), "Ofuscado: mayúscula": ("prostituTa", ""), "Abreviado": ("prosti", ""),
}

st.title("🛡️ Menos odio a la vista · v5")
st.caption("Un único modelo entrega tres señales independientes sobre el comentario y, si existe, su contexto. "
           "Son probabilidades calibradas, no un veredicto: la decisión final corresponde a una persona.")

ej = st.selectbox("Ejemplos", list(EJEMPLOS))
if ej != st.session_state.get("_ej"):
    st.session_state["_ej"] = ej
    st.session_state["txt"], st.session_state["ctx"] = EJEMPLOS[ej]
texto = st.text_area("Comentario", key="txt", height=90)
ctx = st.text_input("Contexto (opcional: título de la noticia o mensaje previo)", key="ctx")
c1, c2 = st.columns(2)
tta = c1.checkbox("Usar vista normalizada (TTA)", value=True, help="Promedia el texto original y su versión sin ofuscación (espaciado, leet, máscaras…).")
comparar = c2.checkbox("Comparar con y sin contexto", value=True)

if st.button("Analizar", type="primary") and texto.strip():
    ctxs = [ctx.strip() or None]
    pc = P.probs([texto], ctxs, tta=tta)[0]
    ps = P.probs([texto], None, tta=tta)[0] if (comparar and ctxs[0]) else None
    norm = R.normalizar_ofuscacion(texto, minusculas=False)
    if norm != texto: st.info(f"Vista normalizada del comentario: **{norm}**")
    for j, h in enumerate(HEADS):
        st.subheader(NOMBRES[h]); st.caption(AYUDA[h])
        if not P.activas[j]:
            st.warning("Cabeza sin entrenar: faltan datos etiquetados para esta categoría."); continue
        p, t = float(pc[j]), float(P.thr[j])
        st.progress(min(max(p, 0.0), 1.0), text=f"probabilidad {p:.3f}  ·  umbral {t:.3f}  →  {'🔴 señal' if p >= t else '🟢 sin señal'}")
        if ps is not None: st.caption(f"Sin contexto: {float(ps[j]):.3f} (cambio al añadir contexto: {p - float(ps[j]):+.3f})")
    with st.expander("Detalles del modelo"):
        st.json({k: v for k, v in P.meta.items()})
        st.write("Umbrales:", dict(zip(HEADS, np.round(P.thr, 3))), "· Temperaturas:", dict(zip(HEADS, np.round(P.T, 3))))
elif not texto.strip():
    st.write("Escriba un comentario o elija un ejemplo.")

with st.expander("Cómo leer el resultado y límites"):
    st.markdown("""
- Las tres categorías **no son excluyentes**: un comentario puede ser vulgar e insultante a la vez.
- El umbral de cada categoría se eligió en una partición de desarrollo y **no es universal**; cambiarlo mueve el equilibrio entre falsos positivos y falsos negativos.
- El **contexto** puede cambiar la señal, pero las categorías de vulgaridad e insulto no se entrenaron con contexto etiquetado: el efecto es una observación, no una garantía.
- Los estereotipos regionales (por ejemplo, contra costeños) **no siempre se reconocen** como odio grupal: el modelo se entrenó con español rioplatense y comentarios de redes de influencers.
- Con textos muy ofuscados el desempeño baja; la vista normalizada ayuda, pero no resuelve todos los casos.
- La herramienta no sustituye la moderación humana ni es una evaluación clínica.
""")

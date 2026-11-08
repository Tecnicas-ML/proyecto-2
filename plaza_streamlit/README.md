# LA PLAZA (Streamlit)

Demo del blog con moderación del modelo **v5 multitarea**, para desplegar en **Streamlit Community Cloud** (sin cupo Gradio de HF).

El comentario se evalúa con el título de la nota como contexto. Se bloquean `insulto` y `odio_grupal`; la vulgaridad sola se publica. Para cambiar la regla, editar `BLOQUEAN` en `streamlit_app.py`.

`multitarea_es.py` y `robustez_es.py` son copias exactas de los módulos del cuaderno.

## Local

Desde la raíz del repo, con un entorno virtual activo:

```bash
pip install -r plaza_streamlit/requirements.txt
streamlit run plaza_streamlit/streamlit_app.py
```

Para usar pesos locales en lugar del Hub, definir `MODELO_DIR` con la ruta de `modelo_v5/` (del `export_v5.zip` del cuaderno).

## Streamlit Cloud

1. Subí estos archivos al repo GitHub (`git push`).
2. Entrá a https://share.streamlit.io → **New app**
3. Repo: `Tecnicas-ML/proyecto-2` (o el tuyo)
4. Branch: `main`
5. Main file path: `plaza_streamlit/streamlit_app.py`
6. Deploy

La primera carga descarga el modelo del Hub (~1–2 min). Requiere que los pesos ya estén publicados; ver «Publicar el modelo v5» en el README principal. Si el repo de pesos es privado o tiene otro nombre, definir `HF_TOKEN` o `HF_MODEL_ID` en los *Secrets* de la app.

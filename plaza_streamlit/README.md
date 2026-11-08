# LA PLAZA (Streamlit)

Demo del blog con moderación del modelo **v5 multitarea**, para desplegar en **Streamlit Community Cloud**. Incluye el laboratorio del modelo (sección 6.3 del cuaderno) como segunda página.

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
4. Branch: `Fine-Tuning2` (o `main`, una vez fusionada)
5. Main file path: `plaza_streamlit/streamlit_app.py`
6. Deploy

La primera carga descarga el modelo del Hub (~1–2 min). Si el repo de pesos cambia de nombre o pasa a privado, definir `HF_MODEL_ID` o `HF_TOKEN` en los *Secrets* de la app.

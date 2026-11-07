# LA PLAZA (Streamlit)

Demo del blog con moderación A3, para desplegar en **Streamlit Community Cloud** (sin cupo Gradio de HF).

## Local

```bash
cd /Users/tatisvivas/Documents/GitHub/proyecto-2
source .venv/bin/activate
pip install -r plaza_streamlit/requirements.txt
streamlit run plaza_streamlit/streamlit_app.py
```

## Streamlit Cloud

1. Subí estos archivos al repo GitHub (`git push`).
2. Entrá a https://share.streamlit.io → **New app**
3. Repo: `TatisVivas/proyecto-2` (o el tuyo)
4. Branch: `main`
5. Main file path: `plaza_streamlit/streamlit_app.py`
6. Deploy

La primera carga descarga el modelo del Hub (~1–2 min).

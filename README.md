# Menos odio a la vista

Detección de comentarios con odio en español. Proyecto de Aplicación 2 (Hugging Face) · Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana.

> **Aviso de contenido:** el corpus y el cuaderno contienen lenguaje ofensivo real.

## Entregable principal

**`proyecto_toxicidad_es.ipynb`** — informe técnico (Fase 0, EDA, preparación, 3 arquitecturas + línea base, errores, costos, app).

## Demos en vivo

| Rol | URL |
|---|---|
| **Laboratorio** (probar el modelo: umbral, probabilidad) | https://huggingface.co/spaces/tatisvivass/menos-odio-a-la-vista |
| **LA PLAZA** (blog con modelo A3 — Streamlit) | https://proyecto-2-eybyvirfnmkjnzroq5qy43.streamlit.app |
| **LA PLAZA estático** (HTML; reglas locales; el Gradio puede fallar) | https://huggingface.co/spaces/tatisvivass/la-plaza-demo · app https://tatisvivass-la-plaza-demo.static.hf.space |
| Pesos A3 | https://huggingface.co/tatisvivass/menos-odio-a3-robertuito |
| Extensión de navegador | Equipo |

Para la demo del pitch preferí **Streamlit** (modelo en proceso). El Space Gradio de HF quedó como **laboratorio**. El estático es respaldo visual: sin Python debajo, depende de reglas JS (+ API si el Gradio responde).

## Repo

```
proyecto_toxicidad_es.ipynb   ← entregable
blog/                         ← HTML estático + assets
plaza_streamlit/              ← app Streamlit (LA PLAZA)
space/                        ← laboratorio Gradio
README.md
.gitignore
```

## Datos y modelo

- Corpus: `piuba-bigdata/contextualized_hate_speech` (etiqueta `HATEFUL`).
- Evaluación principal: partición por artículo (`split_articulo`, seed 42).
- Modelo: **RoBERTuito** fine-tuned (A3), umbral ~0.28 (DEV). F1 odio test ≈ 0,653.

## Cuaderno

Abrir el `.ipynb` (local o Colab GPU T4), ejecutar en orden. Con `TRAIN_A3=False` se reutilizan los pesos del Hub.

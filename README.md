# Menos odio a la vista

Detección de comentarios con odio en español. Proyecto de Aplicación 2 (Hugging Face) · Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana.

> **Aviso de contenido:** el corpus y el cuaderno contienen lenguaje ofensivo real.

## Entregable principal

**`proyecto_toxicidad_es.ipynb`** — informe técnico (Fase 0, EDA, preparación, 3 arquitecturas + línea base, errores, costos, app).

| Extra (demo) | Dónde |
|---|---|
| Blog de prueba | `blog/index.html` (modo simulado o API del Space) |
| App Gradio | https://huggingface.co/spaces/tatisvivass/menos-odio-a-la-vista |
| App en vivo | https://tatisvivass-menos-odio-a-la-vista.hf.space |
| Pesos A3 | https://huggingface.co/tatisvivass/menos-odio-a3-robertuito |
| Extensión de navegador | Equipo |

## Repo (limpio)

```
proyecto_toxicidad_es.ipynb   ← entregable
blog/index.html               ← demo del pitch
space/                        ← código del Space (pesos en el Hub)
README.md
.gitignore
```

Los scripts `.py` locales y los pesos (`a3_model/`) quedan fuera de git. El modelo vive en Hugging Face Hub.

## Datos y modelo elegido

- Corpus: `piuba-bigdata/contextualized_hate_speech` (etiqueta `HATEFUL`).
- Evaluación principal: partición por artículo.
- Modelo seleccionado: **RoBERTuito** fine-tuned (A3), umbral ~0.28 (elegido en DEV).
- Entrenamiento reportado: Apple Silicon (MPS), ~55 min.

## Cómo trabajar el cuaderno

Abrir el `.ipynb` (local o Colab GPU T4), ejecutar en orden. Con `TRAIN_A3=False` se reutilizan los pesos ya entrenados.

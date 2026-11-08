---
language:
- es
tags:
- text-classification
- multi-label-classification
- hate-speech
- spanish
- robertuito
base_model: pysentimiento/robertuito-base-uncased
datasets:
- piuba-bigdata/contextualized_hate_speech
- SINAI/OffendES
metrics:
- f1
- roc_auc
---

# menos-odio-v5-multitarea

RoBERTuito compartido con tres cabezas independientes y no excluyentes: `vulgaridad`, `insulto` y `odio_grupal`. Acepta un contexto opcional (título de la noticia o mensaje previo) y entrega probabilidades calibradas por cabeza.

No es un modelo estándar de `transformers`: se guarda y se carga con la clase `Predictor` de `multitarea_es.py`, incluida en el Space y en el cuaderno del proyecto.

## Datos
- **`odio_grupal`:** `piuba-bigdata/contextualized_hate_speech` (`HATEFUL`), partición por artículo, seed 42. Español rioplatense, tema pandémico.
- **`vulgaridad` e `insulto`:** `SINAI/OffendES` (comentarios de influencers).
- Cada fuente solo entrena las cabezas que anota (BCE con máscara). Ningún conjunto de entrenamiento es colombiano.

## Entrenamiento
- **Dónde:** GPU de Colab, ≈ 13 min (778 s), noviembre de 2026.
- **Hiperparámetros:** 4 épocas, batch 16 con acumulación 2, lr 2e-5, max length 128, seed 42. Mejor checkpoint en la época 3.
- **Aumentación:** perturbaciones ortográficas, vista normalizada y descarte de contexto.
- **Preprocesado:** `pysentimiento.preprocessing.preprocess_tweet(lang="es")`.
- **Calibración:** temperatura por cabeza (0,90 · 1,43 · 1,35), ajustada solo en DEV_EVAL.

## Métricas (DEV_EVAL; TEST sin evaluar)
| Cabeza | n | Positivos | Precision | Recall | F1 | AUC | Umbral |
|---|---:|---:|---:|---:|---:|---:|---:|
| vulgaridad | 37 | 7 | 1,000 | 1,000 | 1,000 | 1,000 | 0,302 |
| insulto | 48 | 11 | 0,900 | 0,818 | 0,857 | 0,961 | 0,351 |
| odio_grupal | 4.517 | 548 | 0,754 | 0,664 | 0,706 | 0,932 | 0,824 |

Vulgaridad e insulto tienen muy pocas filas: **estas cifras no son evidencia de desempeño**.

Comparación con [`tatisvivass/menos-odio-a3-robertuito`](https://huggingface.co/tatisvivass/menos-odio-a3-robertuito) en las mismas filas: F1 0,706 frente a 0,689, ΔF1 = +0,016, IC 95 % [−0,012; 0,045]. **Sin diferencia detectable.**

## Limitaciones
- Los estereotipos regionales colombianos no se reconocen de forma fiable: “Todos los costeños son perezosos” obtiene 0,42 en `odio_grupal`, por debajo del umbral 0,824.
- Vulgaridad e insulto no se entrenaron con contexto etiquetado: el efecto del contexto en esas cabezas es una observación por confirmar.
- Marca como vulgares algunos usos educativos, periodísticos o negados.
- No sustituye la moderación humana.

## Licencia
Pendiente de verificar: la ficha de `SINAI/OffendES` declara licencias contradictorias y estos pesos derivan en parte de ese conjunto.

## Uso
```python
from huggingface_hub import snapshot_download
from multitarea_es import Predictor, HEADS

pred = Predictor.cargar(snapshot_download("DavidVargas-Ctrl/menos-odio-v5-multitarea"))
probs = pred.probs(["Usted es un hp"], ["Discusión acalorada por una deuda"])[0]
print(dict(zip(HEADS, probs)))
```

Space demo: https://huggingface.co/spaces/tatisvivass/menos-odio-a-la-vista

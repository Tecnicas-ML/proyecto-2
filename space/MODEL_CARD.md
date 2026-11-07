---
language:
- es
license: mit
library_name: transformers
tags:
- text-classification
- hate-speech
- spanish
- robertuito
base_model: pysentimiento/robertuito-base-uncased
datasets:
- piuba-bigdata/contextualized_hate_speech
metrics:
- f1
- roc_auc
pipeline_tag: text-classification
---

# menos-odio-a3-robertuito

Fine-tuning de `pysentimiento/robertuito-base-uncased` sobre `piuba-bigdata/contextualized_hate_speech`
(etiqueta binaria `HATEFUL`), partición por artículo (seed 42).

## Entrenamiento
- **Dónde:** Apple Silicon (MPS), Mac local — ~55 min (~3283 s), no Colab.
- **Cuándo:** noviembre 2026
- **Hiperparámetros:** 3 épocas máx., early stopping, batch 16, lr 2e-5, max length 128, seed 42.
- **Preprocesado:** `pysentimiento.preprocessing.preprocess_tweet(lang="es")`

## Métricas (test por artículo)
| Métrica | Valor |
|---|---|
| F1 hateful | 0.653 |
| Precision | 0.686 |
| Recall | 0.623 |
| ROC-AUC | 0.898 |
| Umbral (DEV) | ≈0.28 |

## Uso
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("tatisvivass/menos-odio-a3-robertuito")
model = AutoModelForSequenceClassification.from_pretrained("tatisvivass/menos-odio-a3-robertuito")
```

Space demo: https://huggingface.co/spaces/tatisvivass/menos-odio-a-la-vista

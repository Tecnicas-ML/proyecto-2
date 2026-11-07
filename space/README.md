---
title: Menos odio a la vista
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Filtro previo de odio en español (RoBERTuito A3)
---

# Menos odio a la vista

Clasificador binario de odio en español (`HATEFUL`) con RoBERTuito fine-tuned (A3).

- Umbral por defecto ≈ **0.28** (máx. F1 en DEV)
- F1 odio en test ≈ **0.653** · AUC ≈ **0.90**
- Pesos: [`tatisvivass/menos-odio-a3-robertuito`](https://huggingface.co/tatisvivass/menos-odio-a3-robertuito)

**Límites:** español, textos cortos, no sustituye moderación humana, puede fallar fuera de dominio.

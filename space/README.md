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
short_description: Vulgaridad, insulto y odio grupal en español con contexto
---

# Menos odio a la vista · v5

Un único modelo (RoBERTuito multitarea) entrega tres señales independientes sobre un comentario en español y, si existe, su contexto: **vulgaridad**, **insulto** y **odio grupal**.

- Umbrales por defecto, elegidos en DEV_EVAL: vulgaridad **0,302** · insulto **0,351** · odio grupal **0,824**
- Odio grupal: F1 ≈ **0,706** · AUC ≈ **0,93** en DEV_EVAL, sin diferencia detectable frente a A3 en las mismas filas (0,689)
- Vulgaridad e insulto se validaron con muy pocas filas (37 y 48): sus cifras no permiten concluir nada
- TEST sin evaluar
- Pesos: [`tatisvivass/menos-odio-v5-multitarea`](https://huggingface.co/tatisvivass/menos-odio-v5-multitarea)

**API para el blog:** `POST /gradio_api/call/predict` con `{"data": [comentario, contexto]}`. Devuelve `{estado, mensaje, senales}`, con `prob`, `umbral` y `senal` por categoría.

**Límites:** español, textos cortos, no sustituye moderación humana. Los estereotipos regionales no siempre se reconocen como odio grupal, y el efecto del contexto en vulgaridad e insulto es una observación por confirmar.

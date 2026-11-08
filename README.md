# Menos odio a la vista

Detección multitarea de **vulgaridad, insulto y odio grupal** con contexto, en español. Proyecto de Aplicación 2 (Hugging Face) · Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana.

**Integrantes:** Esteban Salazar, Samuel Pico, David Vargas y Tatiana Vivas.

> **Aviso de contenido:** el corpus y el cuaderno contienen lenguaje ofensivo real.

## Entregable principal

**`proyecto_toxicidad_es_v5.ipynb`** — informe técnico reproducible (Fase 0, EDA, preparación de etiquetas, modelo multitarea, calibración, robustez, contexto, comparación con A3, errores, HateCheck-es, aplicación, limitaciones y conclusiones).

## Demos en vivo

| Rol | URL |
|---|---|
| **Laboratorio** (tres señales, umbrales, efecto del contexto) | https://huggingface.co/spaces/tatisvivass/menos-odio-a-la-vista |
| **LA PLAZA** (blog con el modelo en proceso — Streamlit) | https://proyecto-2-eybyvirfnmkjnzroq5qy43.streamlit.app |
| **LA PLAZA estático** (HTML; llama al laboratorio; sin él usa reglas locales) | https://huggingface.co/spaces/tatisvivass/la-plaza-demo · app https://tatisvivass-la-plaza-demo.static.hf.space |
| Pesos v5 | https://huggingface.co/tatisvivass/menos-odio-v5-multitarea |
| Pesos A3 (modelo original; el cuaderno lo usa como referencia, no borrar) | https://huggingface.co/tatisvivass/menos-odio-a3-robertuito |
| Extensión de navegador | Equipo |

> **Estado del despliegue:** estos enlaces siguen sirviendo el modelo A3 hasta completar los pasos de [Publicar el modelo v5](#publicar-el-modelo-v5).

Para la demo del pitch preferí **Streamlit** (modelo en proceso). El Space Gradio de HF quedó como **laboratorio**. El estático es respaldo visual: sin Python debajo, depende de la API del laboratorio y, si no responde, de reglas JS.

**Regla de moderación de LA PLAZA:** se bloquean `insulto` y `odio_grupal`; la vulgaridad sola se publica. El título de la nota se envía como contexto. Las reglas locales cubren amenazas y exclusión, que el modelo no modela; la lista de insultos solo actúa si el modelo no responde.

## Repo

```
proyecto_toxicidad_es_v5.ipynb   ← entregable
blog/                            ← HTML estático + assets
plaza_streamlit/                 ← app Streamlit (LA PLAZA)
space/                           ← laboratorio Gradio + ficha del modelo
README.md
.gitignore
```

`multitarea_es.py` y `robustez_es.py`, dentro de `space/` y `plaza_streamlit/`, son copias exactas de los módulos que escribe el cuaderno (sección 3.4). Si cambian las celdas `%%writefile`, hay que volver a copiarlos.

## Datos y modelo

- **Odio grupal:** `piuba-bigdata/contextualized_hate_speech` (etiqueta `HATEFUL`), partición por artículo (seed 42).
- **Vulgaridad e insulto:** `SINAI/OffendES` (dataset protegido). Cada fuente solo entrena las cabezas que anota (pérdida con máscara).
- **Modelo:** RoBERTuito compartido + tres cabezas independientes, con contexto opcional, calibración por temperatura y umbrales por cabeza (vulgaridad 0,302 · insulto 0,351 · odio grupal 0,824).

| Cabeza (DEV_EVAL) | n | Positivos | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| vulgaridad | 37 | 7 | 1,000 | 1,000 | 1,000 | 1,000 |
| insulto | 48 | 11 | 0,900 | 0,818 | 0,857 | 0,961 |
| **odio_grupal** | 4.517 | 548 | 0,754 | 0,664 | **0,706** | 0,932 |

Cómo leer estas cifras:

- **Solo `odio_grupal` tiene un tamaño suficiente.** Las cifras de vulgaridad e insulto no permiten concluir nada.
- **Frente a A3 no hay diferencia detectable:** en las mismas filas, 0,706 frente a 0,689 (ΔF1 = +0,016; IC 95 % [−0,012; 0,045]). El valor de v5 está en las dos señales nuevas, la tolerancia a variantes ortográficas y el contexto.
- **El TEST sigue sin evaluar** (`EVALUAR_TEST_FINAL = False`). El F1 ≈ 0,653 reportado antes para A3 es de TEST y no se compara con estas cifras de DEV_EVAL.
- **Punto débil conocido:** los estereotipos regionales colombianos (“Todos los costeños son perezosos”) no activan `odio_grupal`; en la batería, v5 rinde peor que A3 (F1 0,22 frente a 0,50, con 8 positivos y etiquetas sin validar).

## Cuaderno

Abrir el `.ipynb` en Colab con GPU T4 y ejecutar en orden. Requiere aceptar las condiciones de `SINAI/OffendES` en el Hub y guardar `HF_TOKEN` en los *Secrets* de Colab; sin eso, vulgaridad e insulto quedan sin entrenar. `MODO='prueba'` verifica que todo corre con una submuestra; `MODO='completo'` entrena con todos los datos (≈ 13 min).

## Publicar el modelo v5

Los pesos se producen en Colab (`RUN/export_v5/modelo_v5`) y las apps los descargan del Hub, así que **primero van los pesos**.

1. Al final de la ejecución del cuaderno, con un token de escritura de la cuenta dueña del repo:

   ```python
   from huggingface_hub import HfApi
   REPO = "tatisvivass/menos-odio-v5-multitarea"
   api = HfApi(token=HF_TOKEN)
   api.create_repo(REPO, exist_ok=True)
   api.upload_folder(repo_id=REPO, folder_path=str(RUN / "export_v5" / "modelo_v5"))
   ```

2. Subir `space/MODEL_CARD.md` a ese repo con el nombre `README.md`.
3. Subir al Space del laboratorio: `app.py`, `requirements.txt`, `README.md`, `multitarea_es.py` y `robustez_es.py`, todos de `space/`.
4. Subir `blog/index.html` al Space estático `la-plaza-demo`.
5. Fusionar en la rama que despliega Streamlit Cloud (`main`).

Si el repo de pesos cambia de nombre, basta con definir `HF_MODEL_ID` en cada app. Si es privado, también `HF_TOKEN` (en los *Secrets* del Space y de Streamlit).

# Menos odio a la vista

Detección de comentarios con odio en español con aprendizaje profundo. Proyecto de Aplicación 2 (Hugging Face) de Técnicas de Aprendizaje de Máquina, Pontificia Universidad Javeriana.

> **Aviso de contenido:** el corpus y los ejemplos del cuaderno contienen lenguaje ofensivo, discriminatorio y violento real.

## Qué es

Un clasificador que recibe un comentario en español y devuelve la probabilidad de que contenga discurso de odio. Se usa como filtro previo de moderación: ocultar el comentario con un aviso de normas de la comunidad, o enviarlo a revisión humana. No reemplaza la decisión de una persona.

| Componente | Descripción | Estado |
|---|---|---|
| Cuaderno de Colab | Informe técnico: Fase 0, EDA, preparación, 3 arquitecturas, línea base, errores, costos | En progreso |
| App web (Gradio en Hugging Face Spaces) | Cuadro de texto que devuelve la probabilidad de odio | Pendiente |
| Blog de prueba | Página estática donde un comentario agresivo se reemplaza por el aviso de normas | Pendiente |
| Extensión de navegador | Aplica el mismo modelo a páginas externas | Pendiente |

## Estructura

```
proyecto_toxicidad_es.ipynb   Cuaderno principal (entregable)
scripts/smoke_eda_prep.py     Prueba local de las secciones 2 a 4 del cuaderno
scripts/train_a1_mlp.py           Entrena A1 (MLP + TF-IDF)
scripts/train_a2_bilstm.py        Entrena A2 (BiLSTM)
scripts/train_a3_robertuito.py    Fine-tuning A3 (RoBERTuito); usar --fast para humo
scripts/eval_baseline_l.py        Línea base sin ajuste (robertuito-hate-speech)
scripts/make_comparison_table.py  Tabla A1/A2/A3/L
app/gradio_app.py                 App local con el modelo A3
blog/index.html                   Blog demo (ocultar comentario + normas)
artifacts/                        Salidas locales (ignorado por git)
```

## Datos

**Corpus principal:** [`piuba-bigdata/contextualized_hate_speech`](https://huggingface.co/datasets/piuba-bigdata/contextualized_hate_speech). Son tuits argentinos en respuesta a noticias de cinco medios durante la pandemia de COVID-19, anotados por personas. La etiqueta objetivo es `HATEFUL`.

| Partición oficial | Filas | % con odio | Artículos únicos |
|---|---|---|---|
| train | 36.420 | 15,2 % | 990 |
| dev | 9.106 | 15,2 % | 989 |
| test | 11.343 | 15,8 % | 248 |

Cifras verificadas al ejecutar el cuaderno / `scripts/smoke_eda_prep.py` (nov 2026).

**Fuga por artículo (`body`):** el **test oficial no comparte artículos con train** (0 %). El **dev oficial sí: 100 %** de sus artículos ya aparecen en train. Aun así usamos una partición por artículo (`split_articulo_*`) como evaluación principal para que train/dev/test no compartan noticias. Todos los modelos se entrenan y evalúan con esos mismos archivos.

**Datos complementarios (solo evaluación, nunca entrenamiento):** `Paul/hatecheck-spanish` para el análisis de errores y un corpus de otro dominio para la prueba fuera de dominio.

## Modelos

| # | Rol | Modelo |
|---|---|---|
| A1 | Entrenado | MLP sobre TF-IDF de palabras y bigramas |
| A2 | Entrenado | BiLSTM/GRU con vocabulario y embeddings propios |
| A3 | Entrenado | [`pysentimiento/robertuito-base-uncased`](https://huggingface.co/pysentimiento/robertuito-base-uncased) con fine-tuning |
| L | Línea base sin ajuste | [`pysentimiento/robertuito-hate-speech`](https://huggingface.co/pysentimiento/robertuito-hate-speech) |

RoBERTuito exige preprocesar el texto con `pysentimiento.preprocessing.preprocess_tweet` antes del tokenizador y admite hasta 128 tokens.

## Cómo ejecutar

### En Colab

1. Abrir `proyecto_toxicidad_es.ipynb` en Colab con GPU (Entorno de ejecución → Cambiar tipo → T4).
2. Ejecutar las celdas **en orden**: la sección 4 usa variables y funciones definidas en las secciones 2 y 3.
3. Después de crear la partición (sección 3.3), copiar los `.parquet` a la carpeta compartida de Drive y ajustar `SPLIT_DIR` en la celda 4.1.

### En local

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install datasets transformers accelerate evaluate scikit-learn pysentimiento tensorflow torch nltk pyarrow
python scripts/smoke_eda_prep.py
```

El script reproduce el EDA y la preparación fuera de Colab, guarda la partición y `prep_config.json` en `artifacts/` y termina imprimiendo `SMOKE OK`. Descarga el corpus y el tokenizador de Hugging Face, así que necesita internet.

### App y blog (demo)

```bash
source .venv/bin/activate
python app/gradio_app.py          # http://127.0.0.1:7860
open blog/index.html              # modo simulado; o marcar "Usar API real"
```

Modelo elegido: **A3 RoBERTuito** (F1 odio en test 0.653). Umbral por defecto ≈ 0.28 (elegido en DEV).

## Reglas del equipo

- **El test no se toca** hasta la evaluación final. Las decisiones se toman con dev.
- **Una sola partición** para los cuatro modelos: los archivos `split_articulo_*`.
- **No mezclar datasets en el entrenamiento:** cada uno define "odio" o "toxicidad" de forma distinta.
- **Ninguna cifra entra al informe sin una celda ejecutada que la produzca.** Lo dicho por una IA se verifica antes de presentarlo como hallazgo.
- **La app usa el mismo preprocesamiento que el cuaderno**, leído de `prep_config.json`.

## Limitaciones conocidas

- El corpus es de dialecto rioplatense y sobre temas de la pandemia. El modelo puede rendir peor con otros países y otros temas.
- La línea base se ajustó con otros datos y quizá con otra definición de odio. Hay que declararlo al compararla.
- RoBERTuito y el corpus vienen del mismo grupo de investigación y ambos son tuits, así que puede haber solapamiento con el preentrenamiento.

## Licencias y citas

La ficha del corpus no declara licencia. Hay que verificarla antes de redistribuir datos o modelos derivados.

```bibtex
@article{perez2022contextual,
  author  = {Pérez, Juan Manuel and Luque, Franco M. and Zayat, Demian and Kondratzky, Martín and Moro, Agustín and Serrati, Pablo Santiago and Zajac, Joaquín and Miguel, Paula and Debandi, Natalia and Gravano, Agustín and Cotik, Viviana},
  journal = {IEEE Access},
  title   = {Assessing the Impact of Contextual Information in Hate Speech Detection},
  year    = {2023},
  volume  = {11},
  pages   = {30575-30590},
  doi     = {10.1109/ACCESS.2023.3258973}
}

@inproceedings{perez-etal-2022-robertuito,
  title     = {{R}o{BERT}uito: a pre-trained language model for social media text in {S}panish},
  author    = {Pérez, Juan Manuel and Furman, Damián Ariel and Alonso Alemany, Laura and Luque, Franco M.},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year      = {2022},
  pages     = {7235--7243}
}
```

## Equipo

_(completar)_

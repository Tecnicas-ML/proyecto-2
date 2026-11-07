"""LA PLAZA — Streamlit (look de medio + moderación A3).

Local:
  streamlit run plaza_streamlit/streamlit_app.py

Cloud: Main file path = plaza_streamlit/streamlit_app.py
"""
from __future__ import annotations

import re

import streamlit as st
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "tatisvivass/menos-odio-a3-robertuito"
MAX_LEN = 128
MAX_CHARS = 500
DEFAULT_THR = 0.28
ASSET = "https://tatisvivass-la-plaza-demo.static.hf.space/assets"

SEED = [
    {
        "author": "Andrés M.",
        "text": "Qué alegría ver la plaza otra vez abierta. Ojalá mantengan la limpieza los fines de semana.",
    },
    {
        "author": "Laura G.",
        "text": "Bien por la obra, pero falta sombra y más bancas para adultos mayores.",
    },
    {
        "author": "Vecino del centro",
        "text": "No estoy de acuerdo con sacar a todos los vendedores. Hay que regular, no castigar.",
    },
    {
        "author": "Camila R.",
        "text": "Hermosa nota. Gracias por contar también lo que pedían las familias del sector.",
    },
]

BLOCK_BODY = (
    "Tu comentario no se publicó. Puede herir a quien lee —en especial a chicas, niños, "
    "adolescentes y comunidades vulnerables—. Rechazamos insultos, amenazas, discursos de "
    "odio e incitación a hacerse daño. Reformulá y volvé a intentar."
)

INSULTOS = re.compile(
    r"\b(mierda|put[oa]s?|marica[s]?|mariquita[s]?|idiota[s]?|imb[eé]cil(?:es)?|"
    r"est[uú]pido[s]?|basura[s]?|pelotudo[s]?|boludo[s]?|huev[oó]n(?:es)?|"
    r"pendejo[s]?|malparido[s]?|gonorrea[s]?|carechimba[s]?|culicagado[s]?|"
    r"zorra[s]?|perra[s]?|hp|hdp|hijodeputa|hij[oa]s?\s+de\s+puta|la\s+concha|"
    r"carajo|forro[s]?|tarado[s]?|retrasado[s]?|in[uú]til(?:es)?|asco|"
    r"negro[s]?\s+de\s+mierda|sudaca[s]?|indocumentado[s]?)\b",
    re.I,
)
DANO = re.compile(
    r"\b(suicidat[eé]|matat[eé]|matate|morit[eé]|mejor\s+mor[ií]|"
    r"ojal[aá]\s+te\s+mueras|ojal[aá]\s+te\s+mates|anda\s+a\s+morir|"
    r"kill\s+yourself|te\s+voy\s+a\s+matar|voy\s+a\s+matar|"
    r"los\s+voy\s+a\s+matar|ojal[aá]\s+bombardeen|muerte\s+a)\b",
    re.I,
)
ODIO = re.compile(
    r"\b(no\s+deber[ií]an\s+existir|ojal[aá]\s+desaparezcan|"
    r"hay\s+que\s+echar\s+a\s+todos|fuera\s+del\s+pa[ií]s\s+ya|"
    r"extermin(?:ar|io)|limpiar\s+el\s+barrio\s+de|gays?\s+fuera|"
    r"lesbianas?\s+fuera|solo\s+sirven\s+para|son\s+pecadoras?)\b",
    re.I,
)


def soft_normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"([a-záéíóúñü])[^a-záéíóúñü0-9\s]+(?=[a-záéíóúñü])", r"\1", t, flags=re.I)
    t = (
        t.replace("0", "o")
        .replace("1", "i")
        .replace("3", "e")
        .replace("4", "a")
        .replace("5", "s")
        .replace("@", "a")
    )
    t = re.sub(r"pvtas?", "puta", t)
    return t.replace("pvto", "puto")


def rules_block(text: str) -> bool:
    t = soft_normalize(text)
    return bool(DANO.search(t) or ODIO.search(t) or INSULTOS.search(t))


@st.cache_resource(show_spinner="Cargando modelo A3 (primera vez puede tardar)…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    clf = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
    )
    clf.eval()
    return tokenizer, clf


def score_hate(texto: str) -> tuple[bool, float]:
    tokenizer, clf = load_model()
    prep = preprocess_tweet(texto, lang="es")
    enc = tokenizer(prep, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    with torch.no_grad():
        prob = float(torch.softmax(clf(**enc).logits[0], dim=-1)[1].item())
    return prob >= DEFAULT_THR, prob


st.set_page_config(
    page_title="LA PLAZA",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,700&display=swap');
  .stApp {{ background: #fafafa !important; color: #121212 !important; }}
  .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] {{
    color: #121212 !important;
  }}
  [data-testid="stCaption"] {{ color: #3d3d3d !important; }}
  h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: #121212 !important;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 0 !important; max-width: 72rem; }}
  .lp-top {{ background:#111 !important; color:#eee !important; font-size:.72rem; padding:.45rem 0;
    font-family:"Libre Franklin",system-ui,sans-serif; margin: 0 -1rem; padding-left:1rem; padding-right:1rem; }}
  .lp-top span {{ color:#eee !important; }}
  .lp-brand {{ background:#fff; border-bottom:3px solid #e10600; padding:.75rem 0;
    margin: 0 -1rem; padding-left:1rem; padding-right:1rem;
    display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
  .lp-brand img {{ height: 3rem; border-radius: 4px; }}
  .lp-pill {{ font-family:"Libre Franklin",system-ui,sans-serif; font-size:.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.06em; border:1px solid #e2e2e2;
    padding:.3rem .55rem; border-radius:999px; color:#444 !important; }}
  .lp-kicker {{ color:#e10600 !important; font-family:"Libre Franklin",system-ui,sans-serif;
    font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
  .lp-title {{ font-family:"Source Serif 4",Georgia,serif; font-size:clamp(1.7rem,4vw,2.5rem);
    line-height:1.12; font-weight:700; margin:.35rem 0 .7rem; color:#121212 !important; }}
  .lp-by {{ font-family:"Libre Franklin",system-ui,sans-serif; font-size:.86rem; color:#3d3d3d !important;
    padding-bottom:1rem; border-bottom:1px solid #e2e2e2; margin-bottom:1rem; }}
  .lp-lead {{ font-family:"Source Serif 4",Georgia,serif; font-size:1.15rem; line-height:1.55;
    font-weight:600; color:#121212 !important; }}
  .lp-body {{ font-family:"Source Serif 4",Georgia,serif; font-size:1.05rem; line-height:1.7;
    color:#121212 !important; }}
  .lp-body strong {{ color:#121212 !important; }}
  .lp-pull {{ border-left:3px solid #e10600; padding-left:.85rem; font-size:1.2rem; font-weight:600;
    font-family:"Source Serif 4",Georgia,serif; margin:1rem 0; color:#121212 !important; }}
  .lp-comments {{ background:#fff; border:1px solid #e2e2e2; border-top:3px solid #e10600;
    padding:1.1rem 1.15rem 1.25rem; margin-top:1.25rem; }}
  .lp-cmt {{ border-top:1px solid #e2e2e2; padding:.85rem 0; }}
  .lp-cmt .who {{ font-family:"Libre Franklin",system-ui,sans-serif; font-weight:700; font-size:.86rem;
    margin:0 0 .25rem; color:#121212 !important; }}
  .lp-cmt .body {{ font-family:"Source Serif 4",Georgia,serif; margin:0; line-height:1.5;
    color:#121212 !important; }}
  div[data-testid="stForm"] {{ border:0; padding:0; }}
  .stButton>button {{ background:#e10600 !important; color:#fff !important; border:0 !important;
    border-radius:2px !important; font-family:"Libre Franklin",system-ui,sans-serif !important; font-weight:700 !important; }}
  .stButton>button:hover {{ background:#b00000 !important; color:#fff !important; border:0 !important; }}
</style>
<div class="lp-top"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem">
  <span>Bogotá · demo académica</span><span>Medio ficticio · Menos odio a la vista</span>
</div></div>
<div class="lp-brand">
  <img src="{ASSET}/logo-la-plaza.jpg" alt="LA PLAZA" />
  <span class="lp-pill">Streamlit · demo</span>
</div>
""",
    unsafe_allow_html=True,
)

col_main, col_side = st.columns([2.2, 1], gap="large")

with col_main:
    st.markdown('<p class="lp-kicker">Bogotá · Comunidad</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lp-title">La plaza del centro reabrió entre música, puestos y vecinos que volvieron a encontrarse</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="lp-by">Por Camila Restrepo · 6 de noviembre 2026 · Moderación con RoBERTuito (A3)</p>',
        unsafe_allow_html=True,
    )
    st.image(f"{ASSET}/hero-plaza.jpg", use_container_width=True)
    st.caption("La plaza del centro, el día de la reapertura.")
    st.markdown(
        '<p class="lp-lead">Después de ocho meses de obras, el espacio público volvió a llenarse. '
        "Familias, comerciantes y adolescentes coincidieron en pedir que la plaza se mantenga segura y abierta.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="lp-pull">“Internet también es patio: un comentario agresivo llega antes que cualquier moderador.”</p>',
        unsafe_allow_html=True,
    )
    st.image(f"{ASSET}/inline-ciudad.jpg", use_container_width=True)
    st.caption("El centro recupera iluminación y bancas nuevas tras la obra.")
    st.markdown(
        '<p class="lp-body">Si querés opinar, dejá un comentario abajo. Antes de publicarse, un filtro '
        "automático revisa el texto. Si detecta odio o agresión, <strong>el comentario no se envía</strong>.</p>",
        unsafe_allow_html=True,
    )

with col_side:
    st.markdown("### Más noticias")
    for img, title, meta in [
        (f"{ASSET}/thumb-bus.jpg", "Refuerzo de rutas de bus para el fin de semana", "Bogotá · hace 2 h"),
        (f"{ASSET}/thumb-politica.jpg", "Consejo discute presupuesto de cultura 2027", "Política · hace 4 h"),
        (f"{ASSET}/thumb-soccer.jpg", "Clásico local termina en empate", "Deportes · hace 5 h"),
        (f"{ASSET}/thumb-cine.jpg", "Festival de cine al aire libre", "Cultura · hace 7 h"),
    ]:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            st.markdown(f"**{title}**")
            st.caption(meta)

if "comments" not in st.session_state:
    st.session_state.comments = list(SEED)
if "banner" not in st.session_state:
    st.session_state.banner = ""

st.markdown('<div class="lp-comments">', unsafe_allow_html=True)
n = len(st.session_state.comments)
h1, h2 = st.columns([3, 1])
with h1:
    st.subheader("Comentarios")
with h2:
    st.markdown(
        f"<p style='text-align:right;color:#5c5c5c;font-family:Libre Franklin,system-ui;"
        f"font-weight:600;padding-top:.6rem'>{n} comentarios</p>",
        unsafe_allow_html=True,
    )

if st.session_state.banner:
    st.markdown(st.session_state.banner, unsafe_allow_html=True)

with st.form("comment_form", clear_on_submit=True):
    author = st.text_input("Nombre", placeholder="Nombre")
    text = st.text_area("Comentario", placeholder="Escribí tu comentario…", max_chars=MAX_CHARS, height=120)
    submitted = st.form_submit_button("Publicar")
    st.caption("Al publicar aceptás las normas de la comunidad. Máx. 500 caracteres.")

if submitted:
    text = (text or "").strip()
    author = (author or "").strip() or "Anónimo"
    if not text:
        st.session_state.banner = "<p style='color:#5c5c5c'>Escribí un comentario.</p>"
    else:
        with st.spinner("Revisando con el filtro…"):
            try:
                by_rules = rules_block(text)
                by_model, _p = score_hate(text)
                block = by_rules or by_model
            except Exception as exc:  # noqa: BLE001
                # Si el modelo no carga (RAM), al menos aplicamos reglas
                by_rules = rules_block(text)
                block = by_rules
                if not by_rules:
                    st.session_state.banner = (
                        "<div style='background:#fff2f1;border:1px solid #f0b8b3;color:#8a1c16;"
                        "padding:.85rem;border-radius:2px;font-family:Libre Franklin,system-ui'>"
                        "<strong>No se pudo moderar</strong><br/>"
                        f"El modelo no respondió ({type(exc).__name__}). Probá de nuevo en unos segundos."
                        "</div>"
                    )
                    st.rerun()

        if block:
            st.session_state.banner = (
                "<div style='background:#fff2f1;border:1px solid #f0b8b3;color:#8a1c16;"
                "padding:.85rem;border-radius:2px;font-family:Libre Franklin,system-ui'>"
                "<strong>Violación de las normas de la comunidad</strong><br/>"
                f"{BLOCK_BODY}</div>"
            )
        else:
            st.session_state.comments = [{"author": author, "text": text}, *st.session_state.comments]
            st.session_state.banner = (
                "<div style='background:#eef8f0;border:1px solid #b9dfc4;color:#1b5c34;"
                "padding:.85rem;border-radius:2px;font-family:Libre Franklin,system-ui'>"
                "<strong>Comentario publicado</strong><br/>"
                "Tu mensaje quedó visible bajo la nota.</div>"
            )
    st.rerun()

for item in st.session_state.comments:
    st.markdown(
        f"<div class='lp-cmt'><p class='who'>{item['author']}</p>"
        f"<p class='body'>{item['text']}</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Opciones de demo (equipo)"):
    if st.button("Restablecer comentarios"):
        st.session_state.comments = list(SEED)
        st.session_state.banner = ""
        st.rerun()

st.markdown(
    """
<div style="background:#111;color:#bbb;padding:1.25rem 1rem;margin:2rem -1rem 0;font-family:Libre Franklin,system-ui;font-size:.78rem">
  <strong style="color:#fff">LA PLAZA</strong> — medio ficticio · proyecto «Menos odio a la vista»
  (Pontificia Universidad Javeriana). Filtro: RoBERTuito A3 + reglas del sitio.
</div>
""",
    unsafe_allow_html=True,
)

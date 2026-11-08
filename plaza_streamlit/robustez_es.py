# -*- coding: utf-8 -*-
"""
robustez_es.py — utilidades para detección de lenguaje ofensivo en español:
robustez a ofuscación ortográfica, batería contrastiva, métricas, calibración e incertidumbre.

Diseño:
  * Sin dependencia de torch/transformers (solo numpy, pandas, scikit-learn), para poder
    probarse con pytest fuera de la GPU.
  * El normalizador NO clasifica ni expande abreviaturas ambiguas: ("hp", "HP") se dejan intactas.
    Produce una *vista alternativa* del texto; el modelo decide con el contexto.
  * Las etiquetas de la batería son anotaciones PROPUESTAS por los autores del código
    (columna `validado=False`) y deben validarse con el protocolo humano antes de usarse
    como verdad de referencia.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LETRA = r"[^\W\d_]"
# --------------------------------------------------------------------------------------
# 1. NORMALIZACIÓN CONTROLADA (vista alternativa, nunca reemplazo obligatorio)
# --------------------------------------------------------------------------------------
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_HOMOGLIFOS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "τ": "t",
})
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "|": "i"}
_RE_LEET = re.compile(rf"(?<={LETRA})[013457@$!|](?={LETRA})")
_STOP2 = {"de", "la", "el", "en", "un", "se", "me", "te", "lo", "no", "si", "es", "ya", "su", "mi", "tu",
          "al", "le", "ni", "yo", "tú", "él", "ha", "he", "ve", "va", "da", "di", "ir", "ay", "oh", "eh",
          "y", "a", "o", "u", "e"}
_RE_ESPACIADO = re.compile(
    rf"(?<!{LETRA})(?:({LETRA}{{2}}) )?((?:{LETRA}[ .\-_]){{3,}}{LETRA})(?!{LETRA})")
_RE_MASCARA = re.compile(r"[^\W\d_][\w*#•·]*[*#•·][\w*#•·]*")
_RE_ALARGA = re.compile(rf"({LETRA})\1{{2,}}")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")

# Formas canónicas usadas SOLO para reconstruir palabras enmascaradas ("hijuep*ta").
# No se usa para clasificar ni para expandir abreviaturas ("hp" no se expande).
LEXICO_RECUPERACION = [
    "hijueputa", "hijueputas", "hijoputa", "gonorrea", "malparido", "malparida", "puta", "putas", "puto",
    "putos", "prostituta", "prostitutas", "marica", "maricas", "mierda", "pendejo", "pendeja", "idiota",
    "imbecil", "imbécil", "cabron", "cabrón", "verga", "carajo", "huevon", "huevón", "zorra", "gonorreas",
]


def _leet_token(tok: str) -> str:
    if tok[:1] in "@#" or tok.lower().startswith("http") or _RE_EMAIL.fullmatch(tok):
        return tok
    if len(re.findall(LETRA, tok)) < 2:      # protege números, "COVID19", "5G", etc.
        return tok

    def sub(m):
        c = m.group(0)
        if c in "!|":                         # "Hola!Gracias": minúscula+!+Mayúscula = límite de oración
            prev, nxt = m.string[m.start() - 1], m.string[m.end()]
            if prev.islower() and nxt.isupper():
                return c
        return _LEET[c]

    for _ in range(3):
        nuevo = _RE_LEET.sub(sub, tok)
        if nuevo == tok:
            break
        tok = nuevo
    return tok


def _colapsar_espaciado(t: str) -> str:
    def rep(m):
        pre, run = m.group(1), re.sub(r"[ .\-_]", "", m.group(2))
        pegar_pre = bool(pre) and pre.lower() not in _STOP2     # "HI" se pega; "un", "de"... no
        unido = pre + run if pegar_pre else run
        cola = ""
        # Caso ambiguo: "T A y lo sabes" → la "y" final es una palabra real, no parte de la ofuscada.
        # Se separa solo si quitarla deja una forma conocida y conservarla no.
        if unido[-1].lower() in "yaoeu" and unido.lower() not in LEXICO_RECUPERACION \
                and unido[:-1].lower() in LEXICO_RECUPERACION:
            unido, cola = unido[:-1], " " + unido[-1]
        return ((pre + " ") if (pre and not pegar_pre) else "") + unido + cola
    return _RE_ESPACIADO.sub(rep, t)


def _resolver_mascara(core: str) -> Optional[str]:
    low = core.lower()
    cands = set()
    patron = "".join("[a-zñáéíóúü]" if c in "*#•·" else re.escape(c) for c in low)
    cands |= {f for f in LEXICO_RECUPERACION if re.fullmatch(patron, f)}
    m = re.fullmatch(r"([^*#•·]{4,})[*#•·]{2,}", low)
    if m:
        cands |= {f for f in LEXICO_RECUPERACION if f.startswith(m.group(1)) and len(f) > len(m.group(1))}
    if not cands:
        return None
    corto = min(cands, key=len)
    return corto if all(c.startswith(corto) for c in cands) else None


def _mascaras(t: str) -> str:
    def rep(m):
        r = _resolver_mascara(m.group(0))
        return r if r else m.group(0)
    return _RE_MASCARA.sub(rep, t)
def normalizar_ofuscacion(texto: str, minusculas: bool = True) -> str:
    """Revierte ofuscaciones comunes sin alterar el significado léxico de abreviaturas ambiguas.

    Revierte: separación de letras, sustitución interna por números/símbolos, homoglifos,
    caracteres de ancho cero, alargamiento (3+ repeticiones), máscaras con * resolubles de forma única.
    No revierte: eliminación de caracteres. No expande "hp"/"HP".
    """
    t = unicodedata.normalize("NFKC", texto).translate(_ZW).translate(_HOMOGLIFOS)
    t = _colapsar_espaciado(t)
    t = " ".join(_leet_token(tok) for tok in t.split(" "))
    t = _mascaras(t)
    t = _RE_ALARGA.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if minusculas else t


def vistas_texto(texto: str) -> List[str]:
    """[original, normalizada] si la normalización cambia algo; si no, solo [original]."""
    n = normalizar_ofuscacion(texto, minusculas=False)
    return [texto] if n == texto else [texto, n]
# --------------------------------------------------------------------------------------
# 2. PERTURBACIONES SINTÉTICAS (aumentación; misma tasa en ambas clases)
# --------------------------------------------------------------------------------------
_A_LEET = {"a": ["4", "@"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["5", "$"], "t": ["7"]}
_A_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х"}


def p_espaciado(w, rng):
    return rng.choice([" ", ".", "-", "_"]).join(w)


def p_leet(w, rng):
    idx = [i for i in range(1, len(w) - 1) if w[i].lower() in _A_LEET]
    if not idx:
        return w
    i = int(rng.choice(idx))
    return w[:i] + str(rng.choice(_A_LEET[w[i].lower()])) + w[i + 1:]


def p_mascara(w, rng):
    idx = [i for i in range(1, len(w) - 1) if w[i].lower() in "aeiou"] or list(range(1, len(w) - 1))
    if not idx:
        return w
    i = int(rng.choice(idx))
    return w[:i] + "*" + w[i + 1:]


def p_alargar(w, rng):
    idx = [i for i, c in enumerate(w) if c.lower() in "aeiou"]
    if not idx:
        return w
    i = idx[-1] if rng.random() < 0.6 else int(rng.choice(idx))
    return w[:i + 1] + w[i] * int(rng.integers(2, 6)) + w[i + 1:]


def p_caso(w, rng):
    modo = int(rng.integers(0, 3))
    if modo == 0:
        return w.upper()
    if modo == 1:
        return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(w))
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in w)


def p_borrar(w, rng):
    if len(w) < 4:
        return w
    i = int(rng.integers(1, len(w) - 1))
    return w[:i] + w[i + 1:]


def p_homoglifo(w, rng):
    idx = [i for i, c in enumerate(w) if c.lower() in _A_HOMO]
    if not idx:
        return w
    i = int(rng.choice(idx))
    return w[:i] + _A_HOMO[w[i].lower()] + w[i + 1:]


PERTURBACIONES = {"espaciado": p_espaciado, "leet": p_leet, "mascara": p_mascara, "alargar": p_alargar,
                  "caso": p_caso, "borrar": p_borrar, "homoglifo": p_homoglifo}
# --- Perturbaciones añadidas en v3 -----------------------------------------------------
_TECLADO = {"q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
            "o": "ipl", "p": "ol", "a": "qsz", "s": "adw", "d": "sfe", "f": "dgr", "g": "fht", "h": "gjy",
            "j": "hku", "k": "jli", "l": "kop", "z": "ax", "x": "zc", "c": "xv", "v": "cb", "b": "vn",
            "n": "bm", "m": "n", "ñ": "nl"}
_SUSTITUTOS = {"a": "ou", "e": "ai", "i": "eyl", "o": "au", "u": "ox", "s": "zc", "c": "sk", "b": "v", "v": "b",
               "t": "d", "d": "t", "g": "j", "j": "g", "k": "c", "x": "s", "p": "b", "m": "n", "n": "m"}
# Abreviaturas usadas SOLO como aumentación de entrenamiento (forma canónica -> abreviatura).
# El normalizador jamás las expande: "hp" sigue siendo ambiguo y lo resuelve el contexto.
_ABREV = {"hijueputa": "hp", "hijueputas": "hps", "hijoputa": "hp", "hijodeputa": "hdp", "hijodeputas": "hdps"}


def p_sustituir(w, rng):
    """Cambia UNA letra interna por otra parecida o frecuente en errores (hijueputa -> hijuxputa/hijuepvta)."""
    idx = [i for i in range(1, len(w) - 1) if w[i].lower() in _SUSTITUTOS]
    pool = idx or list(range(1, len(w) - 1))
    if not pool:
        return w
    i = int(rng.choice(pool))
    c = w[i].lower()
    nuevo = str(rng.choice(list(_SUSTITUTOS.get(c, "x"))))
    return w[:i] + (nuevo.upper() if w[i].isupper() else nuevo) + w[i + 1:]


def p_teclado(w, rng):
    """Reemplaza una letra interna por una vecina del teclado QWERTY."""
    idx = [i for i in range(1, len(w) - 1) if w[i].lower() in _TECLADO]
    if not idx:
        return w
    i = int(rng.choice(idx))
    nuevo = str(rng.choice(list(_TECLADO[w[i].lower()])))
    return w[:i] + (nuevo.upper() if w[i].isupper() else nuevo) + w[i + 1:]


def p_sin_vocales(w, rng):
    """Omite vocales internas (prostituta -> prstt, hijueputa -> hjpt). Conserva primera y última letra."""
    if len(w) < 5:
        return w
    nucleo = "".join(c for c in w[1:-1] if c.lower() not in "aeiouáéíóú")
    return w[0] + nucleo + w[-1] if nucleo != w[1:-1] and len(nucleo) >= 1 else w


def p_abrev(w, rng):
    """hijueputa -> hp, etc. Solo si la palabra es una forma canónica conocida; si no, no hace nada."""
    r = _ABREV.get(w.lower())
    if r is None:
        return w
    return r.upper() if w.isupper() else r


_BASE_MEZCLA = ["espaciado", "leet", "mascara", "alargar", "caso", "borrar", "homoglifo", "sustituir", "teclado", "sin_vocales"]


def p_mezcla(w, rng):
    """Aplica 2 o 3 perturbaciones distintas en cadena a la misma palabra (p. ej. teclado + leet + caso)."""
    k = int(rng.integers(2, 4))
    for nombre in rng.choice(_BASE_MEZCLA, size=k, replace=False):
        w = PERTURBACIONES[str(nombre)](w, rng)
    return w
PERTURBACIONES.update({"sustituir": p_sustituir, "teclado": p_teclado, "sin_vocales": p_sin_vocales,
                       "abrev": p_abrev, "mezcla": p_mezcla})

_RE_PALABRA = re.compile(rf"^(\W*)({LETRA}+)(\W*)$")


def perturbar_texto(texto: str, rng: np.random.Generator, p_palabra: float = 0.35,
                    tipos: Optional[Sequence[str]] = None, min_len: int = 4) -> Tuple[str, List[str]]:
    """Perturba palabras alfabéticas (>= min_len letras); omite @usuarios, #hashtags y URLs.
    Garantiza al menos una palabra perturbada si existe alguna elegible."""
    tipos = list(tipos or PERTURBACIONES)
    toks = texto.split(" ")
    elegibles = []
    for i, tk in enumerate(toks):
        m = _RE_PALABRA.match(tk)
        if m and len(m.group(2)) >= min_len and tk[:1] not in "@#" and not tk.lower().startswith("http"):
            elegibles.append(i)
    if not elegibles:
        return texto, []
    elegidos = [i for i in elegibles if rng.random() < p_palabra] or [int(rng.choice(elegibles))]
    ops = []
    for i in elegidos:
        pre, w, post = _RE_PALABRA.match(toks[i]).groups()
        # "abrev" solo aplica a formas canónicas conocidas (hijueputa -> hp): se prefiere cuando es posible y se
        # excluye en el resto. Si una operación no cambia la palabra, se reintenta con otra (sin copias idénticas).
        abreviable = w.lower() in _ABREV and "abrev" in tipos
        pool = [t for t in tipos if t != "abrev"] or ["leet"]
        nombre, nueva = None, w
        for intento in range(6):
            nombre = "abrev" if (abreviable and (intento == 0 and rng.random() < 0.5)) else str(rng.choice(pool))
            nueva = PERTURBACIONES[nombre](w, rng)
            if nueva != w:
                break
        toks[i] = pre + nueva + post
        ops.append(nombre)
    return " ".join(toks), ops


def aumentar_df(df: pd.DataFrame, frac: float, seed: int = 42, col_texto: str = "text",
                col_y: str = "HATEFUL", p_palabra: float = 0.35, tipos=None) -> pd.DataFrame:
    """Copias perturbadas de una fracción ESTRATIFICADA por clase (misma tasa en cada clase), para que
    el modelo no aprenda 'texto perturbado = odio'. La etiqueta se conserva."""
    rng = np.random.default_rng(seed)
    sub = df.groupby(col_y, group_keys=False).sample(frac=frac, random_state=seed).copy()
    res = [perturbar_texto(t, rng, p_palabra, tipos) for t in sub[col_texto]]
    sub["texto_original"] = sub[col_texto].values
    sub[col_texto] = [r[0] for r in res]
    sub["perturbado"] = 1
    sub["ops"] = [",".join(r[1]) for r in res]
    return sub
# --------------------------------------------------------------------------------------
# 3. BATERÍA CONTRASTIVA (DEV para elegir transformaciones/umbrales; TEST para el reporte final)
# --------------------------------------------------------------------------------------
_Y_OFENSIVO = {"neutral": 0, "vulgar_sin_agresion": 2, "insulto": 1, "estereotipo": 1, "odio_grupo": 1, "ambiguo": 2}
_Y_GRUPO = {"neutral": 0, "vulgar_sin_agresion": 0, "insulto": 0, "estereotipo": 1, "odio_grupo": 1, "ambiguo": 2}
CATEGORIAS = list(_Y_OFENSIVO)

_OFUSC_HIJUEPUTA = {"espaciado": "HI J U E P U T A", "leet_1": "H1JUEPUTA", "leet_excl": "H!JUEPUTA",
                    "mascara": "hijuep*ta", "puntos": "h.i.j.u.e.p.u.t.a", "alargado": "hijueputaaaa",
                    "caso_alterno": "hIjUePuTa", "homoglifo": "hijuеputa"}
_OFUSC_PROSTITUTA = {"caso": "prostituTa", "leet": "pr0stitu7a", "mascara": "prostitu*a",
                     "espaciado": "p r o s t i t u t a", "alargado": "prostituuuta", "homoglifo": "prоstituta"}


class _B:
    def __init__(self, prefijo):
        self.p, self.rows, self.n = prefijo, [], 0

    def add(self, familia, fenomeno, categoria, texto, nota="", regional="CO", base_id=None, perturbacion=None):
        self.n += 1
        rid = f"{self.p}-{self.n:03d}"
        self.rows.append(dict(id=rid, familia=f"{self.p}-{familia}", fenomeno=fenomeno, categoria=categoria,
                              y_ofensivo=_Y_OFENSIVO[categoria], y_grupo=_Y_GRUPO[categoria], texto=texto,
                              regional=regional, base_id=base_id, perturbacion=perturbacion, nota=nota,
                              validado=False))
        return rid

    def df(self):
        return pd.DataFrame(self.rows)


def _grupos(b, pares, frames):
    for g, a in pares:
        for fen, cat, plantilla in frames:
            b.add(f"GRP-{g}", fen, cat, plantilla.format(g=g, a=a))


def _ofuscar(b, familia, base, token, variantes, cat):
    rid = b.add(familia, "ofuscacion_base", cat, base)
    for nombre, forma in variantes.items():
        b.add(familia, "ofuscacion", cat, base.replace(token, forma), base_id=rid, perturbacion=nombre)


def _benignos_perturbados(b, familia_pref, bases, semillas):
    for k, base in enumerate(bases):
        rid = b.add(f"{familia_pref}{k}", "ofuscacion_base", "neutral", base)
        for s in semillas:
            rng = np.random.default_rng(1000 * k + s)
            t, ops = perturbar_texto(base, rng, p_palabra=0.5, tipos=["leet", "espaciado", "alargar", "caso", "mascara"])
            if t != base:
                b.add(f"{familia_pref}{k}", "ofuscacion_benigna", "neutral", t, base_id=rid, perturbacion=",".join(ops))
def construir_bateria() -> Dict[str, pd.DataFrame]:
    """Devuelve {'dev': df, 'test': df}. Conjuntos disjuntos por familia y por texto."""
    # ---------------- DEV ----------------
    d = _B("DEV")
    _grupos(d, [("costeños", "perezosos"), ("paisas", "tacaños"), ("venezolanos", "ladrones")], [
        ("geografico", "neutral", "Estoy escribiendo un blog sobre la cultura de los {g} y su gastronomía."),
        ("estereotipo", "estereotipo", "Todos los {g} son {a}."),
        ("estereotipo", "estereotipo", "Los {g} son unos {a}, no se puede confiar en ellos."),
        ("contradiscurso", "neutral", "No está bien decir que todos los {g} son {a}."),
        ("cita_denuncia", "neutral", "Hoy me dijeron que los {g} son {a}; eso es discriminación y lo denuncio."),
        ("negacion", "neutral", "Es falso que los {g} sean {a}; trabajan como cualquier otra persona."),
    ])
    d.add("CHAMO", "coloquial_regional", "ambiguo", "Ese videojuego es cosa de chamos, yo prefiero algo más serio.",
          "'cosa de chamos' ≈ cosa de muchachos; solo despectivo según el contexto", "VE/CO")
    d.add("CHAMO", "geografico", "neutral", "Mis amigos chamos organizaron una fiesta y nos divertimos mucho.", regional="VE/CO")
    d.add("CHAMO", "estereotipo", "odio_grupo", "Esos chamos que llegaron al barrio son puros delincuentes, que se vayan de aquí.", regional="VE/CO")
    d.add("CHAMO", "contradiscurso", "neutral", "Dejen de decir que todo lo que hacen los chamos es cosa de delincuentes.", regional="VE/CO")
    d.add("HP", "marca_homonimo", "neutral", "Compré un portátil HP y la batería dura poco.")
    d.add("HP", "marca_homonimo", "neutral", "Mi impresora HP ya no imprime en color.")
    d.add("HP", "insulto", "insulto", "Ese tipo es un hp, me quitó el puesto en la fila.")
    d.add("HP", "coloquial_amistoso", "ambiguo", "Parce, usted sí es mucho hp jugando fútbol jajaja")
    d.add("HP", "periodistico", "neutral", "El mandatario llamó 'hp' a varios periodistas, según informó el diario.")
    d.add("HP", "homonimo_politico", "ambiguo", "En su discurso dijo 'HP' para referirse a los honorables parlamentarios.",
          "uso documentado en prensa colombiana (El Universal, 2025-04-26)")
    d.add("PROS", "insulto", "insulto", "Eres una prostituta y todo el barrio lo sabe.")
    d.add("PROS", "insulto", "insulto", "Esa prosti me robó el novio y ni vergüenza tiene.")
    d.add("PROS", "educativo", "neutral", "La ONG estudia la prostitución desde la salud pública y los derechos laborales.")
    d.add("PROS", "contradiscurso", "neutral", "No le digas prostituta a nadie; usar esa palabra como insulto es violencia verbal.")
    d.add("PROS", "neutral", "neutral", "El documental muestra con respeto la vida de una trabajadora sexual.")
    d.add("HIJ", "insulto", "insulto", "Usted es un hijueputa, no vuelva por aquí.")
    d.add("HIJ", "coloquial_exclamacion", "vulgar_sin_agresion", "¡Hijueputa, se me cayó el celular!")
    d.add("HIJ", "coloquial_amistoso", "vulgar_sin_agresion", "Qué hijueputa susto me dio, parce, casi me da algo jajaja")
    d.add("HIJ", "educativo", "neutral", "En Colombia 'hijueputa' se usa tanto para insultar como para expresar sorpresa.")
    d.add("HIJ", "negacion", "neutral", "No creo que seas un hijueputa; solo estabas de mal humor.")
    d.add("HIJ", "negacion", "neutral", "Nunca le diría hijueputa a mi mamá ni a nadie.")
    d.add("AR", "coloquial_amistoso", "vulgar_sin_agresion", "Che, boludo, no te olvides de traer el mate.", regional="AR")
    d.add("AR", "insulto", "insulto", "Sos un pelotudo, no entendés nada.", regional="AR")
    d.add("MARICA", "coloquial_amistoso", "ambiguo", "Marica, no me deje plantada hoy, mire que la fiesta es importante.")
    d.add("GAY", "odio", "odio_grupo", "Los gays son una plaga y deberían desaparecer.", regional="ES-LATAM")
    d.add("GAY", "contradiscurso", "neutral", "Defiendo los derechos de los gays y de todas las personas.", regional="ES-LATAM")
    d.add("IRON", "ironia", "ambiguo", "Claro, los paisas son tan generosos que ni el aire comparten, jajaja")
    d.add("INV", "inventada", "ambiguo", "Ese tetranutra no sabe lo que dice.",
          "sin significado documentado; la búsqueda web solo halló la marca TetraNatura (alimento para peces)")
    d.add("INV", "marca_homonimo", "neutral", "El alimento TetraNatura es bueno para los peces del acuario.")
    d.add("ANG", "prestamo", "neutral", "Ese cringe de video me dio pena ajena, qué fail.")
    d.add("ANG", "prestamo", "neutral", "Mi crush me dejó en visto y ahora estoy en modo drama jajaja")
    _ofuscar(d, "OFU-HIJ", "Eres un hijueputa y todos lo saben.", "hijueputa", _OFUSC_HIJUEPUTA, "insulto")
    _ofuscar(d, "OFU-PROS", "Eres una prostituta y no sirves para nada.", "prostituta", _OFUSC_PROSTITUTA, "insulto")
    _benignos_perturbados(d, "OFU-BEN", [
        "La reunión es mañana en la oficina del centro.",
        "Gracias por la receta, quedó deliciosa y a mi familia le encantó.",
        "El partido empieza a las ocho y vamos todos juntos al estadio."], semillas=[1, 2, 3])

    # ---------------- TEST ----------------
    t = _B("TST")
    _grupos(t, [("cachacos", "creídos"), ("pastusos", "bobos"), ("porteños", "arrogantes")], [
        ("geografico", "neutral", "Voy a visitar a mis amigos {g} este fin de semana; dicen que su comida es deliciosa."),
        ("estereotipo", "estereotipo", "Los {g} son siempre {a}, ni vale la pena hablarles."),
        ("estereotipo", "estereotipo", "Qué se puede esperar de unos {g}, todos son {a}."),
        ("contradiscurso", "neutral", "Me molesta que digan que los {g} son {a}; es un estereotipo injusto."),
        ("cita_denuncia", "neutral", "Escuché a alguien decir que los {g} son {a} y le respondí que no se generaliza."),
        ("negacion", "neutral", "Los {g} no son {a} por naturaleza; nadie lo es por su origen."),
    ])
    t.add("PELAO", "coloquial_regional", "ambiguo", "Eso del karaoke es cosa de pelaos, a mí no me llama la atención.")
    t.add("CHAMO", "estereotipo", "odio_grupo", "Los chamos que cruzan la frontera son todos unos vagos y ladrones, que los devuelvan.", regional="VE/CO")
    t.add("CHAMO", "neutral", "neutral", "Conocí unos chamos muy trabajadores en el taller de mi tío.", regional="VE/CO")
    t.add("HP", "marca_homonimo", "neutral", "El cargador HP se dañó y no consigo repuesto en Bogotá.")
    t.add("HP", "marca_homonimo", "neutral", "Mi HP Pavilion trae 16 GB de RAM y funciona perfecto.")
    t.add("HP", "insulto", "insulto", "Nos tocó ver ese partido horrible, el árbitro es un hp.")
    t.add("HP", "insulto", "insulto", "Esa HP de vecina nunca saluda y siempre está criticando a todo el mundo.")
    t.add("HP", "coloquial_amistoso", "ambiguo", "Qué hp de gol el de anoche, parce, no lo puedo creer.")
    t.add("HP", "metalinguistico", "neutral", "Cuando alguien dice 'hp' en una entrevista, unos lo toman como insulto y otros como chiste.")
    t.add("PROS", "insulto", "insulto", "Eres una prostituta y una fresca.")
    t.add("PROS", "insulto", "insulto", "Esa prosti ni se baña y todavía se cree la reina.")
    t.add("PROS", "contradiscurso", "neutral", "Me parece mal que usen 'prostituta' como insulto contra las mujeres.")
    t.add("PROS", "neutral", "neutral", "La película cuenta la historia de una prostituta que lucha por criar a su hija.")
    t.add("PROS", "negacion", "neutral", "Jamás diría que eres una prostituta; respeto mucho a las mujeres.")
    t.add("HIJ", "insulto", "insulto", "Eres un hijueputa y ojalá te pase lo peor.")
    t.add("HIJ", "coloquial_exclamacion", "vulgar_sin_agresion", "¡Hijueputa, otra vez llueve y sin paraguas!")
    t.add("HIJ", "coloquial_amistoso", "vulgar_sin_agresion", "Ese hijueputa partido nos dejó sin aire, parce, jajaja")
    t.add("HIJ", "educativo", "neutral", "El profesor explicó que 'hijueputa' es una contracción de 'hijo de puta'.")
    t.add("HIJ", "negacion", "neutral", "No puedo creer que alguien le diga hijueputa a un niño.")
    t.add("MX", "coloquial_amistoso", "vulgar_sin_agresion", "No manches, güey, ¡qué buen concierto estuvo!", regional="MX")
    t.add("MX", "insulto", "insulto", "Eres un pinche idiota, no sirves para nada.", regional="MX")
    t.add("MARICA", "coloquial_amistoso", "ambiguo", "Marica, no me deje sola hoy, mire que la cita me da miedo.")
    t.add("MARICA", "insulto", "insulto", "Ese marica me tiene harto, no hace nada bien y encima se cree mucho.")
    t.add("MUJ", "estereotipo", "estereotipo", "Las mujeres no sirven para manejar, deberían quedarse en la cocina.", regional="ES-LATAM")
    t.add("MUJ", "contradiscurso", "neutral", "Las mujeres manejan tan bien como cualquiera; ese prejuicio es viejo y falso.", regional="ES-LATAM")
    t.add("MUJ", "odio", "odio_grupo", "Hay que sacar del país a todos los indígenas, son una plaga.", regional="ES-LATAM")
    t.add("IRON", "ironia", "ambiguo", "Sí, claro, los cachacos son súper cálidos, por eso saludan a todo el mundo. Jajaja")
    t.add("INV", "inventada", "ambiguo", "Ese cuchiflantro no sabe nada de fútbol.", "palabra inventada para la batería; sin significado conocido")
    t.add("ANG", "prestamo", "insulto", "Eres un loser y un fracasado, nadie te quiere.")
    t.add("ANG", "prestamo", "neutral", "Ese youtuber hizo un unboxing buenísimo y el streaming estuvo genial.")
    _ofuscar(t, "OFU-HIJ", "Qué pena contigo, hijueputa, siempre nos dejas colgados.", "hijueputa", _OFUSC_HIJUEPUTA, "insulto")
    _ofuscar(t, "OFU-PROS", "Todos saben que eres una prostituta, deja de hacerte la santa.", "prostituta", _OFUSC_PROSTITUTA, "insulto")
    _benignos_perturbados(t, "OFU-BEN", [
        "Mañana tengo cita médica y después paso por el supermercado.",
        "Me encantó el concierto de anoche, la banda tocó muy bien.",
        "El profesor explicó el tema con mucha paciencia y ejemplos."], semillas=[1, 2, 3])
    dev, tst = d.df(), t.df()
    assert not (set(dev.familia) & set(tst.familia)) and not (set(dev.texto) & set(tst.texto))
    return {"dev": dev, "test": tst}
# --------------------------------------------------------------------------------------
# 4. MÉTRICAS, CALIBRACIÓN, UMBRALES E INCERTIDUMBRE
# --------------------------------------------------------------------------------------
def metricas_binarias(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    r0 = tn / (tn + fp) if tn + fp else 0.0
    f10 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f1, f1_clase0=f10,
                f1_macro=(f1 + f10) / 2, accuracy=(tp + tn) / max(len(y), 1))


def elegir_umbral_f1(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    mejores = max(np.unique(p), key=lambda t: (metricas_binarias(y, p >= t)["f1"], -abs(t - 0.5)))
    return float(mejores)


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y).astype(float), np.asarray(p).astype(float)
    ids = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def elegir_banda(y, p, prec_obj: float = 0.80, npv_obj: float = 0.95, min_n: int = 20) -> Tuple[float, float]:
    """(lo, hi) con datos de DEV: p>=hi → 'sí' con precisión>=prec_obj; p<lo → 'no' con VPN>=npv_obj;
    en medio → 'revisar'. Si los objetivos no son alcanzables, colapsa a la banda en el umbral F1."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    u = np.unique(p)
    hi = next((t for t in u if (p >= t).sum() >= min_n and y[p >= t].mean() >= prec_obj), None)
    lo = next((t for t in u[::-1] if (p < t).sum() >= min_n and 1 - y[p < t].mean() >= npv_obj), None)
    f1t = elegir_umbral_f1(y, p)
    hi = f1t if hi is None else float(hi)
    lo = f1t if lo is None else float(lo)
    return (min(lo, hi), hi)


def decision_tres_vias(p, lo: float, hi: float, desacuerdo=None, desacuerdo_max: Optional[float] = None):
    """'no' | 'revisar' | 'sí'. Si las vistas (original/normalizada/MC-dropout) discrepan más que
    `desacuerdo_max`, se envía a revisión aunque la probabilidad media esté fuera de la banda."""
    p = np.asarray(p)
    out = np.where(p >= hi, "sí", np.where(p < lo, "no", "revisar")).astype(object)
    if desacuerdo is not None and desacuerdo_max is not None:
        out[np.asarray(desacuerdo) > desacuerdo_max] = "revisar"
    return out


def combinar_vistas(ps: Sequence[Sequence[float]], modo: str = "media") -> np.ndarray:
    a = np.asarray(ps, dtype=float)
    return {"media": a.mean(0), "max": a.max(0), "min": a.min(0)}[modo]
def evaluar_bateria(df: pd.DataFrame, p, thr: float, objetivo: str = "y_ofensivo") -> dict:
    """Resumen sobre las filas con etiqueta binaria (0/1); las ambiguas (2) se reportan aparte."""
    p = np.asarray(p, dtype=float)
    d = df.assign(prob=p, pred=(p >= thr).astype(int))
    b = d[d[objetivo].isin([0, 1])]
    glob = metricas_binarias(b[objetivo], b.pred)
    por_fen = (b.groupby("fenomeno").apply(lambda g: pd.Series(dict(
        n=len(g), acierto=(g[objetivo] == g.pred).mean(), tasa_pred_positiva=g.pred.mean())), include_groups=False)
        .sort_values("acierto"))
    fam = b.groupby("familia").apply(lambda g: bool((g[objetivo] == g.pred).all()), include_groups=False)
    amb = d[d[objetivo] == 2]
    return dict(global_=glob, por_fenomeno=por_fen, acierto_familia=float(fam.mean()) if len(fam) else np.nan,
                n_familias=int(len(fam)), ambiguos=dict(n=len(amb), tasa_pred_positiva=float(amb.pred.mean()) if len(amb) else np.nan,
                prob_media=float(amb.prob.mean()) if len(amb) else np.nan))


def consistencia_ofuscacion(df: pd.DataFrame, p, thr: float, objetivo: str = "y_ofensivo") -> dict:
    """Para cada variante ofuscada con base_id: retención de recall (positivas) y tasa de FP (negativas)
    respecto de la base, y cambio medio absoluto de probabilidad."""
    p = pd.Series(np.asarray(p, dtype=float), index=df["id"].values)
    v = df[df.base_id.notna()]
    if v.empty:
        return {}
    pb = p.loc[v.base_id].values
    pv = p.loc[v["id"]].values
    y = v[objetivo].values
    pos, neg = y == 1, y == 0
    base_det = pb >= thr
    return dict(
        n_variantes=int(len(v)),
        retencion_recall=float((pv[pos & base_det] >= thr).mean()) if (pos & base_det).any() else float("nan"),
        recall_variantes_pos=float((pv[pos] >= thr).mean()) if pos.any() else float("nan"),
        fp_variantes_neg=float((pv[neg] >= thr).mean()) if neg.any() else float("nan"),
        delta_prob_medio=float(np.abs(pv - pb).mean()),
        por_perturbacion=pd.DataFrame(dict(y=y, det=(pv >= thr), perturbacion=v.perturbacion.values))
        .query("y == 1").groupby("perturbacion").det.mean().sort_values().to_dict())


def bootstrap_ic(y, p, thr: float, B: int = 1000, seed: int = 42, grupos=None, metrica: str = "f1"):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    if grupos is None:
        grupos = np.arange(len(y))
    grupos = np.asarray(grupos)
    ug = np.unique(grupos)
    idx_por = {g: np.where(grupos == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        sel = np.concatenate([idx_por[g] for g in rng.choice(ug, len(ug), replace=True)])
        vals.append(metricas_binarias(y[sel], pred[sel])[metrica])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fragmentacion_subword(tokenizer, textos: Iterable[str]) -> float:
    """Tokens subword por palabra (alto = el tokenizador fragmenta la palabra: señal de término raro)."""
    n_tok = n_pal = 0
    for t in textos:
        n_tok += len(tokenizer.tokenize(t))
        n_pal += max(len(t.split()), 1)
    return n_tok / max(n_pal, 1)
# --------------------------------------------------------------------------------------
# 5. DATOS: particiones por grupo, fuga, acuerdo entre anotadores, mapeo de datasets
# --------------------------------------------------------------------------------------
def separar_por_familia(df: pd.DataFrame, col_grupo: str = "familia", test_size: float = 0.3, seed: int = 42):
    from sklearn.model_selection import GroupShuffleSplit
    g = df[col_grupo].factorize()[0]
    i_a, i_b = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(df, groups=g))
    return df.iloc[i_a], df.iloc[i_b]


def verificar_fuga(sp: Dict[str, pd.DataFrame], col_grupo: str = "title", col_texto: str = "text") -> dict:
    """Los parquet guardados no incluyen `body`; `title` es un sustituto conservador (si los títulos son
    disjuntos, los artículos también). Además cuenta textos exactamente duplicados entre particiones."""
    res = {}
    nombres = list(sp)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            res[f"{a}∩{b}:{col_grupo}"] = len(set(sp[a][col_grupo]) & set(sp[b][col_grupo]))
            res[f"{a}∩{b}:texto_duplicado"] = len(set(sp[a][col_texto]) & set(sp[b][col_texto]))
    return res


def fleiss_kappa(conteos: np.ndarray) -> float:
    """conteos: (n_items, n_categorias) con nº de anotadores por categoría (mismo nº de anotadores por ítem)."""
    c = np.asarray(conteos, dtype=float)
    n = c.sum(1)[0]
    p_j = c.sum(0) / c.sum()
    P_i = ((c ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def acuerdo_anotadores(df: pd.DataFrame, cols: Sequence[str]) -> dict:
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    pares = {f"{a}~{b}": float(cohen_kappa_score(df[a], df[b])) for a, b in combinations(cols, 2)}
    cats = sorted(set(np.unique(df[list(cols)].values)))
    conteos = np.stack([(df[list(cols)].values == c).sum(1) for c in cats], axis=1)
    return dict(cohen_pares=pares, cohen_medio=float(np.mean(list(pares.values()))), fleiss=fleiss_kappa(conteos))


# Mapeo propuesto de OffendES (Plaza-del-Arco et al., 2021) a la taxonomía de 6 niveles.
# Verificar nombres de columnas al cargar (el dataset está protegido: requiere aceptar condiciones en el Hub).
# OFG (ofensivo a un grupo) NO se asume equivalente a «grupo» de piuba hasta verificar su definición (OFG_COMO_GRUPO).
MAPA_OFFENDES = {"NO": "neutral", "NOE": "vulgar_sin_agresion", "OFP": "insulto", "OFG": "ofensivo_sin_especificar"}
# --------------------------------------------------------------------------------------
# 6. ENFOQUE HÍBRIDO (Experimento D): n-gramas de caracteres + regresión logística
# --------------------------------------------------------------------------------------
def _prep_norm(t: str) -> str:
    return normalizar_ofuscacion(t)


def _prep_id(t: str) -> str:
    return t.lower()


def entrenar_char_lr(textos, y, C: float = 4.0, ngram=(2, 5), max_features: int = 300_000,
                     normalizar: bool = True, seed: int = 42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2, sublinear_tf=True,
                          max_features=max_features, preprocessor=_prep_norm if normalizar else _prep_id)
    X = vec.fit_transform(list(textos))
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(X, y)
    return vec, clf


def probs_char_lr(vec, clf, textos) -> np.ndarray:
    return clf.predict_proba(vec.transform(list(textos)))[:, 1]


def mezclar(p_transformer, p_char, w: float) -> np.ndarray:
    return (1 - w) * np.asarray(p_transformer) + w * np.asarray(p_char)


def elegir_peso_mezcla(y_dev, p_tr_dev, p_ch_dev, grid=np.linspace(0, 1, 11)) -> Tuple[float, float, float]:
    """Elige w y umbral SOLO con DEV. Devuelve (w, umbral, F1_dev)."""
    mejor = (0.0, 0.5, -1.0)
    for w in grid:
        pm = mezclar(p_tr_dev, p_ch_dev, w)
        thr = elegir_umbral_f1(y_dev, pm)
        f1 = metricas_binarias(y_dev, pm >= thr)["f1"]
        if f1 > mejor[2]:
            mejor = (float(w), float(thr), float(f1))
    return mejor
# --------------------------------------------------------------------------------------
# 7. CANDIDATOS PARA ANOTACIÓN HUMANA (sin etiquetas): base del conjunto de ejemplos difíciles de entrenamiento
# --------------------------------------------------------------------------------------
_PALABRAS_ANOT = [("un hijueputa", "hijueputa"), ("una prostituta", "prostituta"), ("un hp", "hp"),
                  ("un marica", "marica"), ("una gonorrea", "gonorrea"), ("un huevón", "huevón")]
_FRAMES_PALABRA = [
    "Ese man es {art}, me dañó el día entero.",
    "Parcero, no sea {art}, venga y me ayuda con la mudanza jajaja",
    "En clase discutimos por qué algunas personas usan '{bare}' como insulto.",
    "Me dijeron {art} frente a todos y me dolió; nadie debería insultar así.",
    "No soy {art}, deja de decirme eso.",
    "El comentario del video decía '{bare}' y el moderador lo ocultó.",
]
_GRUPOS_ANOT = [("caleños", "flojos"), ("opitas", "cansones"), ("santandereanos", "groseros"), ("limeños", "mentirosos")]
_FRAMES_GRUPO = [
    "Escribo una crónica sobre la música y la comida de los {g}.",
    "Los {g} son {a}, así son todos.",
    "Está mal decir que los {g} son {a}; cada persona es distinta.",
    "En la discusión alguien soltó que los {g} son {a} y la mitad del chat protestó.",
]


def candidatos_anotacion(seed: int = 42) -> pd.DataFrame:
    """Frases candidatas SIN etiqueta (columna `y_final` vacía) para el protocolo de anotación humana.
    Disjuntas de la batería (palabras/grupos/redacción distintos). La misma palabra aparece en
    varios contextos para que los anotadores juzguen la intención, no la palabra."""
    rng = np.random.default_rng(seed)
    filas = []
    for art, bare in _PALABRAS_ANOT:
        for k, fr in enumerate(_FRAMES_PALABRA):
            filas.append(dict(familia=f"ANOT-PAL-{bare}", frame=f"P{k}", texto=fr.format(art=art, bare=bare), perturbacion=None))
        if len(bare) >= 4:                                            # "hp" es demasiado corta para ofuscarse
            for k in (0, 4):                                          # variantes ofuscadas de dos frames
                base = _FRAMES_PALABRA[k].format(art=art, bare=bare)
                forma = PERTURBACIONES[str(rng.choice(["leet", "espaciado", "alargar", "caso"]))](bare, rng)
                filas.append(dict(familia=f"ANOT-PAL-{bare}", frame=f"P{k}-ofus", texto=base.replace(bare, forma),
                                  perturbacion="objetivo"))
    for g, a in _GRUPOS_ANOT:
        for k, fr in enumerate(_FRAMES_GRUPO):
            filas.append(dict(familia=f"ANOT-GRP-{g}", frame=f"G{k}", texto=fr.format(g=g, a=a), perturbacion=None))
    df = pd.DataFrame(filas).drop_duplicates("texto").reset_index(drop=True)
    df.insert(0, "id", [f"ANOT-{i:03d}" for i in range(len(df))])
    for c in ["anotador_1", "anotador_2", "anotador_3", "y_final", "notas_adjudicacion"]:
        df[c] = ""        # categorías válidas: ver CATEGORIAS
    return df
# --------------------------------------------------------------------------------------
# 8. CANDIDATOS CON CONTEXTO (sin etiquetas): el mismo comentario en contextos de conversación distintos
# --------------------------------------------------------------------------------------
_CONTEXTOS_ANOT = [
    ("amistoso", "Dos amigos hablan por WhatsApp después de ver un partido de fútbol juntos."),
    ("pelea", "Discusión acalorada entre desconocidos en los comentarios de una noticia política."),
    ("denuncia", "Una usuaria cuenta que la insultaron en la calle y pide que no se normalice."),
    ("educativo", "Clase de lingüística sobre expresiones coloquiales del español de Colombia."),
    ("prensa", "Nota de prensa sobre una polémica por unas declaraciones de un funcionario."),
]
_COMENTARIOS_CTX = [
    "Usted sí es {art}, parce, no me joda.",
    "Eso que dijo ese man es de {art}.",
    "Me parece que {art} es lo mínimo que merece.",
    "{bare}, no puedo creer lo que pasó.",
]
_COMENTARIOS_GRUPO_CTX = [
    "Los {g} son {a}, todo el mundo lo sabe.",
    "Dicen que los {g} son {a}.",
]


def candidatos_anotacion_contexto(seed: int = 42) -> pd.DataFrame:
    """Filas (contexto, texto) SIN etiqueta para anotar con el protocolo humano. Cada comentario aparece en varios
    contextos: los anotadores juzgan la intención dada la situación, no la palabra. `familia` agrupa todos los contextos
    de un mismo comentario (partir siempre por familia). Disjuntas de la batería."""
    filas = []
    for art, bare in _PALABRAS_ANOT:
        for k, fr in enumerate(_COMENTARIOS_CTX):
            for nom, ctx in _CONTEXTOS_ANOT:
                filas.append(dict(familia=f"CTX-PAL-{bare}-C{k}", contexto_tipo=nom, contexto=ctx,
                                  texto=fr.format(art=art, bare=bare.capitalize())))
    for g, a in _GRUPOS_ANOT:
        for k, fr in enumerate(_COMENTARIOS_GRUPO_CTX):
            for nom, ctx in _CONTEXTOS_ANOT:
                filas.append(dict(familia=f"CTX-GRP-{g}-C{k}", contexto_tipo=nom, contexto=ctx, texto=fr.format(g=g, a=a)))
    df = pd.DataFrame(filas).drop_duplicates(["contexto", "texto"]).reset_index(drop=True)
    df.insert(0, "id", [f"CTX-{i:03d}" for i in range(len(df))])
    for c in ["anotador_1", "anotador_2", "anotador_3", "y_final", "notas_adjudicacion"]:
        df[c] = ""
    return df
# --------------------------------------------------------------------------------------
# v3 — etiquetas de 4 clases, partición de DEV por artículo, calibración y reporte binario
# --------------------------------------------------------------------------------------
# Una sola cabeza de 4 clases: neutral · vulgar_sin_agresion · insulto · odio_estereotipo_grupo.
#   P(grupo) = P3 (objetivo original, comparable con A3) · P(ofensivo) = P2 + P3
CLASES4 = ["neutral", "vulgar_sin_agresion", "insulto", "odio_estereotipo_grupo"]
NO_GRUPO, OFENSIVO_GEN, IGNORAR = -1, -2, -100
# Códigos de etiqueta por ejemplo:  0..3 = clase completa · -1 = «NO es grupo» (piuba HATEFUL=0), pérdida -log(1-P3)
#   · -2 = «ofensivo sin especificar», pérdida -log(P2+P3) · -100 = sin etiqueta (se ignora).
OFG_COMO_GRUPO = False       # True SOLO si se verificó que OFG de OffendES equivale a HATEFUL de piuba


def codigos_piuba(hateful) -> np.ndarray:
    """HATEFUL=1 → clase grupo (3). HATEFUL=0 → -1: no es grupo; puede ser neutral, vulgar o insulto y NO se asigna."""
    return np.where(np.asarray(hateful).astype(int) == 1, 3, NO_GRUPO)


def codigos_offendes(etiquetas: Sequence[str], ofg_como_grupo: bool = False) -> np.ndarray:
    """OffendES: NO→neutral, NOE→vulgar, OFP→insulto. OFG→«ofensivo sin especificar» salvo verificación previa."""
    m = {"NO": 0, "NOE": 1, "OFP": 2, "OFG": 3 if ofg_como_grupo else OFENSIVO_GEN}
    out = []
    for e in etiquetas:
        if e not in m:
            raise ValueError(f"etiqueta OffendES inesperada: {e!r}")
        out.append(m[e])
    return np.array(out)


def codigos_taxonomia(categorias: Sequence[str]) -> np.ndarray:
    """Taxonomía de 6 niveles anotada por humanos → códigos. «ambiguo» y vacíos se ignoran (los gestiona la banda de «revisar»)."""
    m = {"neutral": 0, "vulgar_sin_agresion": 1, "insulto": 2, "estereotipo": 3, "odio_grupo": 3}
    return np.array([m.get(c, IGNORAR) for c in categorias])
def partir_dev_por_grupo(df: pd.DataFrame, col_grupo: str = "title", frac_minado: float = 0.5, seed: int = 42):
    """Parte DEV por artículo en (dev_minado, dev_eval). Hard-negative mining usa SOLO dev_minado para entrenar;
    umbral, temperatura, banda, checkpoint y comparación se calculan en dev_eval, que ningún modelo ve al entrenar."""
    from sklearn.model_selection import GroupShuffleSplit
    g = df[col_grupo].factorize()[0]
    i_eval, i_min = next(GroupShuffleSplit(1, test_size=frac_minado, random_state=seed).split(df, groups=g))
    return df.iloc[i_min], df.iloc[i_eval]


def minar_errores(y, p, thr: float, max_fp: Optional[int] = None, max_fn: Optional[int] = None):
    """Índices de falsos positivos y negativos, ordenados por la confianza del error. Nunca usar con la batería TEST."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    fp = np.where((y == 0) & (p >= thr))[0]; fn = np.where((y == 1) & (p < thr))[0]
    return fp[np.argsort(-p[fp])][:max_fp], fn[np.argsort(p[fn])][:max_fn]


def logit_binario(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def ajustar_temperatura(z, y) -> float:
    """Temperature scaling sobre un logit binario (T>1 suaviza, T<1 afila). Se ajusta SOLO con DEV."""
    from scipy.optimize import minimize_scalar
    z, y = np.asarray(z, dtype=float), np.asarray(y).astype(float)

    def nll(logT):
        q = np.clip(1 / (1 + np.exp(-z / np.exp(logT))), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def aplicar_temperatura(p, T: float) -> np.ndarray:
    return 1 / (1 + np.exp(-logit_binario(p) / T))


def reporte_binario(y, p, thr: float, bins: int = 10) -> dict:
    """Precisión, recall y F1 por clase y macro, matriz de confusión [[TN, FP], [FN, TP]] y ECE."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int); out = {}
    for c in (0, 1):
        tp = int(((y == c) & (pred == c)).sum()); fp = int(((y != c) & (pred == c)).sum()); fn = int(((y == c) & (pred != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        out[f"clase{c}"] = dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, soporte=int((y == c).sum()))
    out["f1_macro"] = (out["clase0"]["f1"] + out["clase1"]["f1"]) / 2
    out["matriz_confusion"] = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    ids = np.minimum((p * bins).astype(int), bins - 1)
    out["ece"] = float(sum((ids == b).mean() * abs(y[ids == b].mean() - p[ids == b].mean()) for b in range(bins) if (ids == b).any()))
    return out

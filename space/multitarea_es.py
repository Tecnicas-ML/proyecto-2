# -*- coding: utf-8 -*-
"""
multitarea_es.py — arquitectura multitarea (v5) para toxicidad en español.

Un encoder Transformer compartido + TRES cabezas independientes y NO excluyentes:
    vulgaridad · insulto (ataque personal) · odio_grupal
Entrada: comentario + contexto opcional (par de segmentos, truncación solo del contexto).
Pérdida: BCE-con-logits enmascarada (las etiquetas ausentes NO aportan gradiente).
Reutiliza de la v4 (robustez_es.py): normalización de ofuscación, perturbaciones, métricas, calibración, bootstrap.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import robustez_es as R
HEADS = ["vulgaridad", "insulto", "odio_grupal"]
NH = len(HEADS)


def fijar_semilla(s: int = 42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
# ======================================================================================
# 3. ETIQUETAS: matriz Y (N×3, NaN = ausente) y máscara M (True = hay etiqueta compatible)
# ======================================================================================
def _a_matriz(Y: np.ndarray):
    Y = np.asarray(Y, dtype=np.float32)
    M = ~np.isnan(Y)
    return np.where(M, Y, 0.0).astype(np.float32), M


def etiquetas_piuba(hateful) -> tuple:
    """piuba `HATEFUL` define SOLO odio contra grupos. HATEFUL=0 NO implica «sin insulto» ni «sin vulgaridad»
    (los insultos personales aparecen como no-hateful en el corpus) → únicamente supervisa la cabeza odio_grupal."""
    h = np.asarray(hateful).astype(int)
    Y = np.full((len(h), NH), np.nan, dtype=np.float32)
    Y[:, HEADS.index("odio_grupal")] = h
    return _a_matriz(Y)


def etiquetas_offendes(etiquetas: Sequence[str], ofg_como_grupo: bool = False) -> tuple:
    """OffendES (NO/NOE/OFP/OFG). Solo se usa lo que la definición respalda:
       NO  (no ofensivo)            → vulgaridad=0, insulto=0, odio_grupal=0
       NOE (no ofensivo, con vulg.) → vulgaridad=1, insulto=0, odio_grupal=0
       OFP (ofensivo a persona)     → insulto=1           (vulgaridad y grupo: ausentes)
       OFG (ofensivo a grupo)       → odio_grupal=1 SOLO si ofg_como_grupo=True (verificar definición vs piuba)
    Fijar `ofg_como_grupo=True` únicamente tras comparar la guía de anotación de OffendES con la de piuba."""
    nan = np.nan
    tabla = {"NO": (0, 0, 0), "NOE": (1, 0, 0), "OFP": (nan, 1, nan),
             "OFG": (nan, nan, 1 if ofg_como_grupo else nan)}
    filas = []
    for e in etiquetas:
        if e not in tabla:
            raise ValueError(f"etiqueta OffendES inesperada: {e!r}")
        filas.append(tabla[e])
    return _a_matriz(np.array(filas, dtype=np.float32))


# Taxonomía de 6 niveles de la v4 (anotación humana) → cabezas. Vacío = ausente (no se adivina).
_TAXO = {"neutral": (0, 0, 0), "vulgar_sin_agresion": (1, 0, 0), "insulto": (np.nan, 1, 0),
         "estereotipo": (np.nan, np.nan, 1), "odio_grupo": (np.nan, np.nan, 1),
         "ambiguo": (np.nan, np.nan, np.nan)}


def etiquetas_taxonomia(categorias: Sequence[str]) -> tuple:
    return _a_matriz(np.array([_TAXO.get(c, (np.nan,) * NH) for c in categorias], dtype=np.float32))


def etiquetas_anotadas(df: pd.DataFrame) -> tuple:
    """CSV de anotación humana. Prioridad: columnas explícitas y_vulgaridad / y_insulto / y_odio_grupal
    (vacío = ausente; permiten multietiqueta real); si no existen, columna `y_final` con la taxonomía v4."""
    cols = [f"y_{h}" for h in HEADS]
    if any(c in df.columns for c in cols):
        Y = np.stack([pd.to_numeric(df[c], errors="coerce").values if c in df.columns else np.full(len(df), np.nan)
                      for c in cols], axis=1)
        bad = Y[~np.isnan(Y)]
        assert set(np.unique(bad)) <= {0.0, 1.0}, "las etiquetas deben ser 0/1"
        return _a_matriz(Y)
    return etiquetas_taxonomia(df["y_final"].astype(str).tolist())


def auditoria_etiquetas(Y: np.ndarray, M: np.ndarray, fuente: Optional[Sequence[str]] = None, min_pos: int = 50, min_neg: int = 50) -> pd.DataFrame:
    """Tabla de disponibilidad por cabeza (y por fuente). `entrenable` = suficientes positivos y negativos etiquetados."""
    filas = []
    fuente = np.asarray(fuente) if fuente is not None else np.array(["todo"] * len(Y))
    for f in list(np.unique(fuente)) + ["TOTAL"]:
        sel = np.ones(len(Y), bool) if f == "TOTAL" else fuente == f
        for j, h in enumerate(HEADS):
            m = M[sel, j]
            pos, neg = int((Y[sel, j][m] == 1).sum()), int((Y[sel, j][m] == 0).sum())
            filas.append(dict(fuente=f, cabeza=h, n_etiquetados=pos + neg, positivos=pos, negativos=neg,
                              entrenable=(pos >= min_pos and neg >= min_neg) if f == "TOTAL" else None))
    return pd.DataFrame(filas)


def cabezas_activas(Y, M, min_pos=50, min_neg=50) -> List[bool]:
    return [bool((Y[M[:, j], j] == 1).sum() >= min_pos and (Y[M[:, j], j] == 0).sum() >= min_neg) for j in range(NH)]


def pesos_positivos(Y, M, tope: float = 10.0) -> np.ndarray:
    """pos_weight por cabeza = neg/pos entre las filas ETIQUETADAS (tope para estabilidad)."""
    w = np.ones(NH, dtype=np.float32)
    for j in range(NH):
        p, n = (Y[M[:, j], j] == 1).sum(), (Y[M[:, j], j] == 0).sum()
        if p > 0 and n > 0:
            w[j] = min(n / p, tope)
    return w
# ======================================================================================
# 4. PREPROCESAMIENTO + AUMENTACIÓN (mismo código en entrenamiento e inferencia)
# ======================================================================================
def _preproc_basico(t: str) -> str:
    import re
    t = re.sub(r"https?://\S+|www\.\S+", "url", t)
    t = re.sub(r"@\w+", "@usuario", t)
    return re.sub(r"\s+", " ", t).strip()


def obtener_preprocesador():
    """`preprocess_tweet(lang='es')` de pysentimiento (lo espera RoBERTuito); si no está instalado, respaldo básico.
    Devuelve (función, nombre) para dejarlo registrado en la configuración del modelo."""
    try:
        from pysentimiento.preprocessing import preprocess_tweet
        return (lambda t: preprocess_tweet(t, lang="es")), "pysentimiento.preprocess_tweet(es)"
    except Exception:
        return _preproc_basico, "basico(url/@usuario)"


class DatosComentarios(torch.utils.data.Dataset):
    """Cada ítem: (texto, contexto|None, y[3], m[3]).
    Aumentación en línea (solo entrenamiento): con prob. `p_aug` el comentario se perturba (misma tasa en todas las
    filas, sin importar la clase); con prob. `p_vista_norm` se entrega la vista normalizada (la misma que usa la TTA);
    con prob. `p_ctx_drop` se elimina el contexto para que el modelo funcione sin él."""

    def __init__(self, textos, contextos, Y, M, preproc, p_aug=0.0, p_vista_norm=0.0, p_ctx_drop=0.0,
                 tipos=None, p_palabra=0.35, seed=42):
        self.t, self.c = list(textos), (list(contextos) if contextos is not None else [None] * len(textos))
        self.Y, self.M = Y.astype(np.float32), M.astype(bool)
        self.pre, self.p_aug, self.p_norm, self.p_ctx = preproc, p_aug, p_vista_norm, p_ctx_drop
        self.tipos, self.p_pal = tipos, p_palabra
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t, c = self.t[i], self.c[i]
        if self.p_aug and self.rng.random() < self.p_aug:
            t, _ = R.perturbar_texto(t, self.rng, self.p_pal, self.tipos)
        elif self.p_norm and self.rng.random() < self.p_norm:
            t = R.normalizar_ofuscacion(t, minusculas=False)
        if self.p_ctx and self.rng.random() < self.p_ctx:
            c = None
        return self.pre(t), (self.pre(c) if isinstance(c, str) and c.strip() else None), self.Y[i], self.M[i]
# ======================================================================================
# 5. TOKENIZACIÓN CON CONTEXTO (par de segmentos; solo se trunca el contexto)
# ======================================================================================
def codificar(tok, textos, contextos=None, max_len=128, usar_contexto=True, reservado_contexto=24):
    """Con contexto: `<s> comentario </s></s> contexto </s>` con truncation='only_second' (el comentario nunca se
    recorta por el contexto). Un comentario más largo que `max_len-reservado` se acorta antes (token a token) para que
    only_second siempre sea viable. Contexto ausente → segmento vacío (mismo formato que en entrenamiento)."""
    textos = list(textos)
    if not usar_contexto:
        enc = tok(textos, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    else:
        b = [(c if isinstance(c, str) else "") for c in (contextos if contextos is not None else [None] * len(textos))]
        ids = tok(textos, add_special_tokens=False)["input_ids"]
        enc = None
        # `only_second` falla si el comentario solo ya excede max_len (tokens especiales incluidos, o si decode→encode
        # reexpande tokens). Se acorta el comentario con margen creciente hasta que el par sea viable.
        for max_a in (max_len - reservado_contexto, max_len // 2, max_len // 4, 8):
            a = [t if len(i) <= max_a else tok.decode(i[:max_a], skip_special_tokens=True) for t, i in zip(textos, ids)]
            try:
                enc = tok(a, b, truncation="only_second", max_length=max_len, padding=True, return_tensors="pt")
                break
            except Exception:
                continue
        if enc is None:      # último recurso: solo el comentario, recortado
            enc = tok(textos, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    enc.pop("token_type_ids", None)
    return enc


class Colador:
    def __init__(self, tok, max_len=128, usar_contexto=True):
        self.tok, self.max_len, self.uc = tok, max_len, usar_contexto

    def __call__(self, lote):
        t, c, y, m = zip(*lote)
        enc = codificar(self.tok, t, c, self.max_len, self.uc)           # padding dinámico: solo hasta el más largo del lote
        enc["labels"] = torch.tensor(np.stack(y)); enc["mask"] = torch.tensor(np.stack(m))
        return enc
# ======================================================================================
# 6. ARQUITECTURA
# ======================================================================================
class MultitareaToxicidad(nn.Module):
    """Encoder compartido → vector [CLS] (<s>) o media → 3 cabezas independientes → 3 logits (sigmoide por cabeza)."""

    def __init__(self, encoder_name: Optional[str] = None, config=None, dropout: float = 0.1,
                 head_hidden: int = 0, pooling: str = "cls"):
        super().__init__()
        from transformers import AutoModel
        kw = dict(add_pooling_layer=False)
        try:
            self.encoder = AutoModel.from_config(config, **kw) if config is not None else AutoModel.from_pretrained(encoder_name, **kw)
        except TypeError:
            self.encoder = AutoModel.from_config(config) if config is not None else AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.pooling, self.head_hidden, self.dropout_p = pooling, head_hidden, dropout
        self.cabezas = nn.ModuleDict({n: self._cabeza(h, head_hidden, dropout) for n in HEADS})

    @staticmethod
    def _cabeza(h, oculto, p):
        if oculto:
            return nn.Sequential(nn.Dropout(p), nn.Linear(h, oculto), nn.GELU(), nn.Dropout(p), nn.Linear(oculto, 1))
        return nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, **_):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            v = hs[:, 0]
        return torch.cat([self.cabezas[n](v) for n in HEADS], dim=1)       # (B, 3) logits


def perdida_enmascarada(logits, y, m, pos_weight=None, head_w=None):
    """BCE-con-logits por (ejemplo, cabeza) × máscara. Normaliza por nº de etiquetas presentes de cada cabeza y
    promedia las cabezas con al menos una etiqueta en el lote. Sin etiquetas → 0 conectado al grafo (no NaN)."""
    z, y, mf = logits.float(), y.float(), m.float()
    pw = pos_weight.to(z.device) if pos_weight is not None else torch.ones(NH, device=z.device)
    l = nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    l = l * (1 + (pw - 1) * y)                                   # pos_weight solo sobre positivos
    n = mf.sum(0)
    por_cabeza = (l * mf).sum(0) / n.clamp(min=1)
    w = (head_w.to(z.device) if head_w is not None else torch.ones(NH, device=z.device)) * (n > 0).float()
    total = (por_cabeza * w).sum() / w.sum().clamp(min=1e-8) if w.sum() > 0 else z.sum() * 0.0
    return total, por_cabeza.detach()


def verificar_gradientes(model, lote: dict, pos_weight=None) -> dict:
    """forward + pérdida + backward; devuelve estadísticas para comprobar que fluye gradiente a encoder y cabezas."""
    model.train(); model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device
    b = {k: v.to(dev) for k, v in lote.items()}
    z = model(**b)
    loss, ph = perdida_enmascarada(z, b["labels"], b["mask"], pos_weight)
    loss.backward()
    g = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    con = [n for n, x in g.items() if x is not None and torch.isfinite(x).all() and x.abs().sum() > 0]
    return dict(loss=float(loss.detach()), loss_finita=bool(torch.isfinite(loss)), forma_logits=tuple(z.shape),
                params_con_grad=len(con), params_total=len(g),
                grad_cabezas={h: float(sum(p.grad.abs().sum() for p in model.cabezas[h].parameters() if p.grad is not None)) for h in HEADS},
                grad_encoder=float(sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)),
                perdida_por_cabeza=ph.cpu().numpy().round(4).tolist())
# ======================================================================================
# 7-8. ENTRENAMIENTO, VALIDACIÓN, EARLY STOPPING, CHECKPOINT
# ======================================================================================
@dataclass
class CfgEntr:
    epochs: int = 3
    bs: int = 16
    lr: float = 2e-5
    lr_cabezas: Optional[float] = None       # None → igual que el encoder
    wd: float = 0.01
    warmup: float = 0.06
    max_len: int = 128
    grad_clip: float = 1.0
    amp: bool = True                         # fp16 en T4 (solo CUDA)
    acum: int = 1                            # acumulación de gradiente (lote efectivo = bs·acum)
    eval_cada: float = 0.5                   # fracción de época entre validaciones
    paciencia: int = 4                       # nº de validaciones sin mejora
    max_pasos: Optional[int] = None          # corta el entrenamiento (modo prueba)
    p_aug: float = 0.30
    p_vista_norm: float = 0.10
    p_ctx_drop: float = 0.30
    tope_pos_weight: float = 10.0
    usar_contexto: bool = True
    seed: int = 42


@torch.no_grad()
def logits_de(model, tok, textos, contextos, preproc, max_len=128, usar_contexto=True, bs=64, device=None) -> np.ndarray:
    """Logits (N×3) en el orden original; lotes ordenados por longitud para minimizar padding."""
    device = device or next(model.parameters()).device
    model.eval()
    t = [preproc(x) for x in textos]
    c = [preproc(x) if isinstance(x, str) and x.strip() else None for x in (contextos if contextos is not None else [None] * len(t))]
    orden = np.argsort([len(x) + (len(y) if y else 0) for x, y in zip(t, c)])
    out = np.zeros((len(t), NH), dtype=np.float32)
    amp = device.type == "cuda"
    for i in range(0, len(t), bs):
        ix = orden[i:i + bs]
        enc = codificar(tok, [t[j] for j in ix], [c[j] for j in ix], max_len, usar_contexto)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            z = model(**enc)
        out[ix] = z.float().cpu().numpy()
    return out


def puntaje_validacion(Y, M, Z, activas) -> dict:
    """F1 de la clase positiva con el mejor umbral de la propia validación, por cabeza ACTIVA con ambos valores
    presentes; el criterio de checkpoint es el promedio (macro) de esas cabezas."""
    P = 1 / (1 + np.exp(-Z)); res = {}
    for j, h in enumerate(HEADS):
        m = M[:, j]
        if not activas[j] or m.sum() == 0 or len(np.unique(Y[m, j])) < 2:
            continue
        thr = R.elegir_umbral_f1(Y[m, j], P[m, j])
        res[h] = R.metricas_binarias(Y[m, j], P[m, j] >= thr)["f1"]
    return dict(por_cabeza=res, macro=float(np.mean(list(res.values()))) if res else float("nan"))
def entrenar(model, tok, ds_train, val: dict, cfg: CfgEntr, preproc, carpeta: Path, pos_weight=None, activas=None, device=None, log=print):
    """Bucle de entrenamiento. `val` = dict(textos, contextos, Y, M) de DEV_CKPT (nunca TEST ni la partición donde se
    calibra/compara). Guarda `mejor.pt` cada vez que mejora el F1 macro de validación; early stopping por paciencia."""
    from transformers import get_linear_schedule_with_warmup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fijar_semilla(cfg.seed)
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    model.to(device)
    activas = activas or [True] * NH
    dl = torch.utils.data.DataLoader(ds_train, batch_size=cfg.bs, shuffle=True, collate_fn=Colador(tok, cfg.max_len, cfg.usar_contexto),
                                     generator=torch.Generator().manual_seed(cfg.seed))
    hw = torch.tensor([1.0 if a else 0.0 for a in activas])      # cabezas sin datos suficientes no se optimizan
    pw = torch.tensor(pos_weight) if pos_weight is not None else None
    sin_decay = ("bias", "LayerNorm.weight", "layer_norm")
    def grupos(params, lr):
        return [dict(params=[p for n, p in params if not any(s in n for s in sin_decay)], weight_decay=cfg.wd, lr=lr),
                dict(params=[p for n, p in params if any(s in n for s in sin_decay)], weight_decay=0.0, lr=lr)]
    opt = torch.optim.AdamW(grupos(list(model.encoder.named_parameters()), cfg.lr) +
                            grupos(list(model.cabezas.named_parameters()), cfg.lr_cabezas or cfg.lr))
    pasos_ep = math.ceil(len(dl) / cfg.acum)
    total = min(cfg.epochs * pasos_ep, cfg.max_pasos or 10**12)
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)
    amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cada = max(1, int(cfg.eval_cada * pasos_ep))
    hist, mejor, sin_mejora, paso, t0 = [], -np.inf, 0, 0, time.time()
    parar = False

    def validar():
        Z = logits_de(model, tok, val["textos"], val["contextos"], preproc, cfg.max_len, cfg.usar_contexto, device=device)
        s = puntaje_validacion(val["Y"], val["M"], Z, activas)
        pl, _ = perdida_enmascarada(torch.tensor(Z), torch.tensor(val["Y"]), torch.tensor(val["M"]), pw, hw)
        model.train()
        return s, float(pl)

    model.train(); opt.zero_grad(set_to_none=True)
    for ep in range(cfg.epochs):
        acc, n_acc = 0.0, 0
        for k, lote in enumerate(dl):
            lote = {a: b.to(device) for a, b in lote.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z = model(**lote)
            loss, _ = perdida_enmascarada(z, lote["labels"], lote["mask"], pw, hw)
            scaler.scale(loss / cfg.acum).backward()
            acc += float(loss.detach()); n_acc += 1
            if (k + 1) % cfg.acum == 0 or k + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
                paso += 1
                if paso % cada == 0 or paso == total:
                    s, vl = validar()
                    crit = s["macro"] if not np.isnan(s["macro"]) else -vl
                    hist.append(dict(paso=paso, epoca=ep + (k + 1) / len(dl), loss_train=acc / max(n_acc, 1), loss_val=vl,
                                     f1_macro_val=s["macro"], **{f"f1_{h}": v for h, v in s["por_cabeza"].items()},
                                     seg=round(time.time() - t0, 1)))
                    acc, n_acc = 0.0, 0
                    if crit > mejor:
                        mejor, sin_mejora = crit, 0
                        torch.save(model.state_dict(), carpeta / "mejor.pt")
                        hist[-1]["checkpoint"] = True
                    else:
                        sin_mejora += 1
                    log({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in hist[-1].items()})
                    if sin_mejora >= cfg.paciencia:
                        log("early stopping"); parar = True
                if paso >= total:
                    parar = True
            if parar:
                break
        if parar:
            break
    model.load_state_dict(torch.load(carpeta / "mejor.pt", map_location=device))      # se queda con el mejor checkpoint
    pd.DataFrame(hist).to_csv(carpeta / "historial.csv", index=False)
    return pd.DataFrame(hist)
# ======================================================================================
# Persistencia + inferencia (Predictor): el MISMO objeto sirve en notebook, Gradio y HF Spaces
# ======================================================================================
class Predictor:
    def __init__(self, model, tok, preproc, preproc_nombre, max_len=128, usar_contexto=True, activas=None,
                 temperaturas=None, umbrales=None, banda=None, device=None, meta=None):
        self.model, self.tok, self.pre, self.pre_nombre = model, tok, preproc, preproc_nombre
        self.max_len, self.uc = max_len, usar_contexto
        self.activas = list(activas) if activas is not None else [True] * NH
        self.T = list(temperaturas) if temperaturas is not None else [1.0] * NH
        self.thr = list(umbrales) if umbrales is not None else [0.5] * NH
        self.meta = meta or {}
        self.device = device or next(model.parameters()).device
        self.model.to(self.device).eval()

    # --- probabilidades calibradas por cabeza (NaN en cabezas sin entrenar) ---
    def probs(self, textos, contextos=None, tta=True, bs=64) -> np.ndarray:
        textos = list(textos)
        contextos = list(contextos) if contextos is not None else [None] * len(textos)
        vistas, dueno, ctxs = [], [], []
        for i, t in enumerate(textos):
            vs = R.vistas_texto(t) if tta else [t]            # [original] o [original, normalizada]
            vistas += vs; dueno += [i] * len(vs); ctxs += [contextos[i]] * len(vs)
        Z = logits_de(self.model, self.tok, vistas, ctxs, self.pre, self.max_len, self.uc, bs, self.device)
        P = 1 / (1 + np.exp(-Z / np.array(self.T, dtype=np.float32)))
        dueno = np.array(dueno)
        out = np.stack([P[dueno == i].mean(0) for i in range(len(textos))]) if len(textos) else np.zeros((0, NH))
        out[:, [not a for a in self.activas]] = np.nan
        return out

    def predecir(self, textos, contextos=None, tta=True) -> pd.DataFrame:
        P = self.probs(textos, contextos, tta)
        df = pd.DataFrame({"texto": list(textos)})
        for j, h in enumerate(HEADS):
            df[f"p_{h}"] = P[:, j]
            df[h] = np.where(np.isnan(P[:, j]), None, P[:, j] >= self.thr[j]) if self.activas[j] else "sin_entrenar"
        return df

    # --- guardar / cargar ---
    def guardar(self, carpeta):
        carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
        self.model.encoder.config.save_pretrained(carpeta / "encoder_config")
        torch.save(self.model.state_dict(), carpeta / "pesos.pt")
        self.tok.save_pretrained(carpeta / "tokenizer")
        cfg = dict(heads=HEADS, max_len=self.max_len, usar_contexto=self.uc, activas=self.activas, temperaturas=self.T,
                   umbrales=self.thr, preprocesamiento=self.pre_nombre, pooling=self.model.pooling,
                   head_hidden=self.model.head_hidden, dropout=self.model.dropout_p, meta=self.meta)
        (carpeta / "multitarea_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return carpeta

    @classmethod
    def cargar(cls, carpeta, device=None):
        from transformers import AutoConfig, AutoTokenizer
        carpeta = Path(carpeta)
        cfg = json.loads((carpeta / "multitarea_config.json").read_text(encoding="utf-8"))
        model = MultitareaToxicidad(config=AutoConfig.from_pretrained(carpeta / "encoder_config"), dropout=cfg["dropout"],
                                    head_hidden=cfg["head_hidden"], pooling=cfg["pooling"])
        model.load_state_dict(torch.load(carpeta / "pesos.pt", map_location="cpu"))
        pre, nombre = obtener_preprocesador()
        if nombre != cfg["preprocesamiento"]:
            print(f"⚠ preprocesamiento del entrenamiento: {cfg['preprocesamiento']} | disponible ahora: {nombre}")
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, AutoTokenizer.from_pretrained(carpeta / "tokenizer"), pre, cfg["preprocesamiento"], cfg["max_len"],
                   cfg["usar_contexto"], cfg["activas"], cfg["temperaturas"], cfg["umbrales"], device=dev, meta=cfg["meta"])
def calibrar(Z: np.ndarray, Y: np.ndarray, M: np.ndarray, activas) -> dict:
    """Temperatura y umbral F1 por cabeza, SOLO con la partición de calibración (DEV_EVAL). Cabezas inactivas o sin
    ambos valores en la partición quedan con T=1 y umbral 0.5 y se marcan como no calibradas."""
    T, thr, ok = [1.0] * NH, [0.5] * NH, [False] * NH
    for j in range(NH):
        m = M[:, j]
        if not activas[j] or m.sum() < 20 or len(np.unique(Y[m, j])) < 2:
            continue
        p0 = 1 / (1 + np.exp(-Z[m, j]))
        T[j] = R.ajustar_temperatura(R.logit_binario(p0), Y[m, j])
        thr[j] = R.elegir_umbral_f1(Y[m, j], 1 / (1 + np.exp(-Z[m, j] / T[j])))
        ok[j] = True
    return dict(temperaturas=T, umbrales=thr, calibradas=ok)
# ======================================================================================
# 9-10. EVALUACIÓN (multietiqueta enmascarada, robustez, contexto, A3, errores)
# ======================================================================================
def metricas_multietiqueta(Y, M, P, thr, nombres=HEADS) -> pd.DataFrame:
    """Por cabeza (solo filas etiquetadas): P/R/F1 de la clase positiva, F1 de la negativa, F1 macro de la cabeza,
    AUC y matriz de confusión. Última fila: macro entre cabezas evaluables. NO mezcla tareas en una sola cifra."""
    from sklearn.metrics import roc_auc_score
    filas = []
    for j, h in enumerate(nombres):
        m = M[:, j] & ~np.isnan(P[:, j])
        if m.sum() == 0 or len(np.unique(Y[m, j])) < 2:
            filas.append(dict(cabeza=h, n=int(m.sum()), nota="sin etiquetas suficientes en esta partición")); continue
        r = R.metricas_binarias(Y[m, j], P[m, j] >= thr[j])
        filas.append(dict(cabeza=h, n=r["n"], positivos=int(Y[m, j].sum()), precision=r["precision"], recall=r["recall"], f1=r["f1"],
                          f1_macro_cabeza=r["f1_macro"], auc=float(roc_auc_score(Y[m, j], P[m, j])), umbral=thr[j],
                          TN=r["tn"], FP=r["fp"], FN=r["fn"], TP=r["tp"]))
    d = pd.DataFrame(filas)
    ev = d[d.get("f1").notna()] if "f1" in d else d.iloc[0:0]
    if len(ev):
        d = pd.concat([d, pd.DataFrame([dict(cabeza="MACRO (cabezas evaluables)", n=int(ev.n.sum()), f1=ev.f1.mean(),
                                             precision=ev.precision.mean(), recall=ev.recall.mean(), auc=ev.auc.mean())])], ignore_index=True)
    return d


def evaluar_ortografia(pred: Predictor, df: pd.DataFrame, Y, M, tipos=None, seed=42, tta=True, col_ctx=None) -> pd.DataFrame:
    """Perturba el comentario (una familia de ofuscación por vez; las mismas filas en cada fila de la tabla) y mide F1
    por cabeza evaluable frente al texto original. Retención = F1_perturbado / F1_original."""
    tipos = tipos or [k for k in R.PERTURBACIONES if k not in ("abrev",)]
    ctx = df[col_ctx].tolist() if col_ctx else None
    base = metricas_multietiqueta(Y, M, pred.probs(df.text.tolist(), ctx, tta), pred.thr)
    f0 = base.set_index("cabeza").f1
    filas = []
    for tp in tipos:
        rng = np.random.default_rng(seed)
        tx = [R.perturbar_texto(t, rng, 0.5, [tp])[0] for t in df.text]
        m = metricas_multietiqueta(Y, M, pred.probs(tx, ctx, tta), pred.thr).set_index("cabeza").f1
        for h in HEADS:
            if h in m.index and not np.isnan(m[h]) and f0.get(h, np.nan) == f0.get(h, np.nan):
                filas.append(dict(perturbacion=tp, cabeza=h, f1_original=f0[h], f1_perturbado=m[h], retencion=m[h] / f0[h] if f0[h] else np.nan))
    return pd.DataFrame(filas)


def etiquetas_bateria(bat: pd.DataFrame):
    """Etiquetas de la batería v4 por cabeza, derivadas de `categoria` (PROPUESTAS por los autores, validado=False)."""
    return etiquetas_taxonomia(bat.categoria.tolist())


def evaluar_bateria_cabezas(pred: Predictor, bat: pd.DataFrame, tta=True) -> dict:
    Y, M = etiquetas_bateria(bat)
    P = pred.probs(bat.texto.tolist(), None, tta)
    return dict(metricas=metricas_multietiqueta(Y, M, P, pred.thr), P=P)
PRUEBAS_CONTEXTO = [
    # (caso, comentario, contexto_amistoso/neutral, contexto_hostil) — pruebas FUNCIONALES, no etiquetas de entrenamiento
    ("cultura", "Me encanta la cultura costeña", None),
    ("estereotipo", "Todos los costeños son perezosos", None),
    ("contradiscurso", "No está bien decir que todos los costeños son perezosos", None),
    ("hp_amistoso", "Usted sí es mucho hp, parce", "Conversación entre amigos: planeando el asado del domingo, todo en buena onda."),
    ("hp_discusion", "Usted es un hp", "Discusión acalorada: le reclama a otro usuario por una deuda y lo amenaza."),
    ("ofuscado_espaciado", "HI J U E P U T A", None),
    ("ofuscado_leet", "H1JUEPUTA", None),
    ("ofuscado_mascara", "hijuep*ta", None),
    ("ofuscado_caso", "prostituTa", None),
    ("abreviado", "prosti", None),
]


def evaluar_contexto(pred: Predictor, casos=PRUEBAS_CONTEXTO, tta=True) -> pd.DataFrame:
    """Probabilidades por cabeza SIN y CON contexto para cada caso; no hay etiqueta, se inspeccionan y se comparan."""
    t = [c[1] for c in casos]; ctx = [c[2] for c in casos]
    sin, con = pred.probs(t, None, tta), pred.probs(t, ctx, tta)
    filas = []
    for i, c in enumerate(casos):
        f = dict(caso=c[0], comentario=c[1], contexto=c[2] or "—")
        for j, h in enumerate(HEADS):
            f[f"{h}|sin_ctx"] = sin[i, j]; f[f"{h}|con_ctx"] = con[i, j]
        filas.append(f)
    return pd.DataFrame(filas)


def errores_fp_fn(df: pd.DataFrame, y, p, thr: float, k: int = 10, col="text"):
    y, p = np.asarray(y).astype(int), np.asarray(p)
    fp, fn = R.minar_errores(y, p, thr, k, k)
    return df.iloc[fp][[col]].assign(prob=p[fp]), df.iloc[fn][[col]].assign(prob=p[fn])
class ProbA3:
    """Modelo A3 ORIGINAL (clasificación binaria de odio). Solo inferencia; jamás se modifica ni se guarda."""

    def __init__(self, ruta_o_hub: str, device=None, max_len=128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ruta_o_hub)
        self.m = AutoModelForSequenceClassification.from_pretrained(ruta_o_hub).eval()
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m.to(self.dev); self.max_len = max_len
        self.pre, self.pre_nombre = obtener_preprocesador()

    @torch.no_grad()
    def probs(self, textos, bs=64) -> np.ndarray:
        t = [self.pre(x) for x in textos]; orden = np.argsort([len(x) for x in t]); out = np.zeros(len(t), dtype=np.float32)
        for i in range(0, len(t), bs):
            ix = orden[i:i + bs]
            enc = self.tok([t[j] for j in ix], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            out[ix] = torch.softmax(self.m(**enc).logits.float(), -1)[:, 1].cpu().numpy()
        return out


def bootstrap_delta_f1(y, p_a, thr_a, p_b, thr_b, grupos, B=1000, seed=42) -> dict:
    """Diferencia pareada F1(b) − F1(a) con remuestreo por artículo (mismo remuestreo para ambos modelos)."""
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y).astype(int), np.asarray(p_a) >= thr_a, np.asarray(p_b) >= thr_b
    g = np.asarray(grupos); ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    d = []
    for _ in range(B):
        sel = np.concatenate([idx[k] for k in rng.choice(ug, len(ug), replace=True)])
        d.append(R.metricas_binarias(y[sel], pb[sel])["f1"] - R.metricas_binarias(y[sel], pa[sel])["f1"])
    return dict(delta_medio=float(np.mean(d)), ic95=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                prob_mejora=float(np.mean(np.array(d) > 0)))
# ======================================================================================
# 12. EXPORTACIÓN (Gradio / HF Spaces)
# ======================================================================================
APP_GRADIO = '''# -*- coding: utf-8 -*-
import gradio as gr, pandas as pd
from multitarea_es import Predictor, HEADS
P = Predictor.cargar("modelo_v5")
def analizar(comentario, contexto):
    df = P.predecir([comentario], [contexto or None])
    filas = []
    for j, h in enumerate(HEADS):
        v = df.loc[0, f"p_{h}"]
        filas.append([h, "sin entrenar (faltan datos)" if P.activas[j] is False else round(float(v), 3),
                      "—" if not P.activas[j] else ("sí" if v >= P.thr[j] else "no"), round(P.thr[j], 3)])
    return pd.DataFrame(filas, columns=["categoría", "probabilidad", "señal", "umbral"])
demo = gr.Interface(analizar, [gr.Textbox(label="Comentario"), gr.Textbox(label="Contexto (opcional)")],
                    gr.Dataframe(label="Resultado"), title="Detector de toxicidad (v5)",
                    description="Las señales son probabilísticas y requieren revisión humana; no son un veredicto.")
if __name__ == "__main__":
    demo.launch()
'''


def exportar_app(pred: Predictor, destino="export_v5") -> Path:
    import shutil
    d = Path(destino); (d).mkdir(parents=True, exist_ok=True)
    pred.guardar(d / "modelo_v5")
    for f in ("multitarea_es.py", "robustez_es.py"):
        if Path(f).exists():
            shutil.copy(f, d / f)
    (d / "app.py").write_text(APP_GRADIO, encoding="utf-8")
    (d / "requirements.txt").write_text("torch\ntransformers\ngradio\nnumpy\npandas\nscikit-learn\nscipy\nemoji\npysentimiento\n", encoding="utf-8")
    return d
# ======================================================================================
# Utilidades de PRUEBA TÉCNICA sin red (encoder diminuto aleatorio + datos SINTÉTICOS)
# Sirven solo para verificar que el código corre; sus métricas no tienen valor científico.
# ======================================================================================
_PAL_NEUTRAS = "hoy mañana noticia gobierno ciudad precio partido concierto profesor clase tema gente país trabajo casa familia calle bus metro".split()
_PAL_GRUPO = "costeños paisas venezolanos cachacos migrantes indígenas".split()
_PAL_ODIO = "plaga basura ladrones perezosos asquerosos".split()
_PAL_INS = "idiota imbécil estúpido inútil hijueputa gonorrea".split()
_PAL_VULG = "mierda carajo hijueputa gonorrea".split()


def datos_sinteticos(n=600, seed=42, n_titulos=60) -> pd.DataFrame:
    """DataFrame con el esquema de piuba (id, text, title, context_tweet, HATEFUL) y texto generado al azar:
    HATEFUL=1 si la plantilla mezcla un grupo con un término despectivo. Solo para pruebas técnicas."""
    rng = np.random.default_rng(seed); filas = []
    for i in range(n):
        base = list(rng.choice(_PAL_NEUTRAS, rng.integers(4, 9)))
        odio = rng.random() < 0.18
        if odio:
            base += [str(rng.choice(_PAL_GRUPO)), "son", str(rng.choice(_PAL_ODIO))]
        elif rng.random() < 0.2:
            base += ["usted", "es", "un", str(rng.choice(_PAL_INS))]
        elif rng.random() < 0.2:
            base += [str(rng.choice(_PAL_VULG)), "qué", "día"]
        rng.shuffle(base)
        t = int(rng.integers(0, n_titulos))
        filas.append(dict(id=i, text=" ".join(base), title=f"titular {t} " + " ".join(rng.choice(_PAL_NEUTRAS, 5)),
                          context_tweet="", HATEFUL=int(odio)))
    return pd.DataFrame(filas)


def datos_offendes_sinteticos(n=300, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed); etq = rng.choice(["NO", "NOE", "OFP", "OFG"], n, p=[.4, .2, .25, .15])
    tx = [" ".join(rng.choice(_PAL_NEUTRAS + (_PAL_VULG if e == "NOE" else _PAL_INS if e == "OFP" else _PAL_GRUPO if e == "OFG" else []), 6)) for e in etq]
    return pd.DataFrame(dict(text=tx, label=etq))


def construir_encoder_minimo(textos: Sequence[str], carpeta, hidden=64, capas=2, max_pos=130):
    """BPE a nivel de byte entrenado localmente + RoBERTa diminuto SIN preentrenar. Devuelve (tokenizer, config)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaTokenizerFast
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer(); bpe.train_from_iterator(list(textos), vocab_size=600, min_frequency=1,
                                                          special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(carpeta))
    tok = RobertaTokenizerFast.from_pretrained(str(carpeta))      # (en transformers 5 `vocab_file=` crea un tokenizador vacío)
    assert len(tok) > 50, 'tokenizador vacío'
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=hidden, num_hidden_layers=capas, num_attention_heads=2,
                        intermediate_size=hidden * 2, max_position_embeddings=max_pos + 2, pad_token_id=tok.pad_token_id,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, type_vocab_size=1)
    return tok, cfg

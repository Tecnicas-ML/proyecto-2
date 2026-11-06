"""Arma la tabla comparativa A1/A2/A3/L a partir de los JSON en artifacts/."""
from __future__ import annotations

import json
import re
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
FILES = {
    "A1 MLP + TF-IDF": "a1_metrics.json",
    "A2 BiLSTM": "a2_metrics.json",
    "A3 RoBERTuito (elegido)": "a3_metrics.json",
    "L robertuito-hate-speech": "baseline_l_metrics.json",
}


def pr_from_report(report: str) -> tuple[float | None, float | None]:
    # línea de la clase 1 / 1.0
    for line in report.splitlines():
        if re.match(r"\s*1(\.0)?\s+", line):
            parts = line.split()
            return float(parts[1]), float(parts[2])
    return None, None


rows = []
for label, fname in FILES.items():
    d = json.loads((ART / fname).read_text())
    tm = d["test_metrics"]
    p, r = pr_from_report(tm.get("report", ""))
    rows.append(
        {
            "modelo": label,
            "F1_hateful": round(tm["f1_hateful"], 3),
            "Precision": round(p, 3) if p is not None else None,
            "Recall": round(r, 3) if r is not None else None,
            "ROC_AUC": round(tm["roc_auc"], 3),
            "umbral_dev": round(tm.get("thr", d.get("dev_metrics", {}).get("thr", 0.5)), 3),
            "train_s": d.get("train_seconds") or d.get("train_search_seconds"),
            "latencia_ms": d.get("latency_ms_per_example_est"),
            "dispositivo": d.get("device", "cpu/mps"),
        }
    )

(ART / "comparison_table.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

md = [
    "# Comparación en test (`split_articulo`, seed 42)",
    "",
    "| Modelo | F1 odio | Precision | Recall | ROC-AUC | Umbral (dev) | Tiempo ent. (s) | Latencia (ms) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md.append(
        f"| {r['modelo']} | {r['F1_hateful']} | {r['Precision']} | {r['Recall']} | {r['ROC_AUC']} | "
        f"{r['umbral_dev']} | {r['train_s']} | {r['latencia_ms']} |"
    )
md += [
    "",
    "**Decisión:** A3 RoBERTuito por mejor F1/AUC en test. La línea base L usa otra definición (multietiqueta) y queda abajo.",
    "",
    "Máquina local: Apple Silicon (MPS) para A3 y L; A1/A2 en CPU/TF.",
]
text = "\n".join(md)
(ART / "comparison_table.md").write_text(text, encoding="utf-8")
print(text)
print("\nguardado artifacts/comparison_table.{json,md}")

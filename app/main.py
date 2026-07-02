"""
AutoInsight API — BernoulliNB (Cristian)
========================================
Plantilla lista para que tu modelo BernoulliNB se conecte al dashboard
grupal de Exequiel (https://api-autoinsight.vercel.app).

Lo único que tenés que hacer:
  1. Poner tus 2 artefactos joblib en `models/`:
       - bernoulli_nb.joblib
       - bernoulli_nb_metadata.joblib
  2. Si tu metadata tiene otros nombres de campos, ajustá la sección
     "Cargar artefactos" abajo. El resto del archivo NO toca tu modelo.
  3. Ejecutar local:  uvicorn app.main:app --reload --port 8000
  4. Deploy donde sea (Easypanel, Render, Fly.io). CORS ya está abierto.

El dashboard auto-detecta tu API porque expone:
  /classes              -> activa el panel "Clases del modelo"
  /probabilities        -> activa el panel "Probabilidades P(feature|clase)"
  /metrics              -> activa el panel "Métricas + matriz confusión"
  /predict              -> permite hacer predicciones desde el formulario
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import joblib
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Cargar artefactos (ajustá las rutas si tus joblib se llaman distinto)
# ──────────────────────────────────────────────────────────────
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
try:
    model    = joblib.load(f"{MODELS_DIR}/bernoulli_nb.joblib")
    metadata = joblib.load(f"{MODELS_DIR}/bernoulli_nb_metadata.joblib")
    FEATURES = metadata["features"]
    # BernoulliNB tiene 2 clases (es_zona_critica_nocturna: si/no)
    CLASSES  = metadata.get("classes") or ["no_critica", "critica"]
    print(f"BernoulliNB cargado | features={len(FEATURES)} | clases={CLASSES}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"No se encontraron los artefactos en {MODELS_DIR}/. "
        f"Asegurate de tener bernoulli_nb.joblib + metadata."
    ) from e

# Colores por clase (crítica/no crítica → rojo/verde)
COLOR_BY_CLASS = {
    "critica":    "#D62828", "Critica":    "#D62828", "CRITICA":    "#D62828",
    "no_critica": "#2D6A4F", "No_critica": "#2D6A4F", "NO_CRITICA": "#2D6A4F",
    "1": "#D62828", "0": "#2D6A4F",
    1:   "#D62828", 0:   "#2D6A4F",
}
COLOR_FALLBACK = ["#2D6A4F", "#D62828"]

def color_de(clase, idx: int) -> str:
    return COLOR_BY_CLASS.get(clase, COLOR_FALLBACK[idx % len(COLOR_FALLBACK)])

# ──────────────────────────────────────────────────────────────
# App + CORS abierto para que el dashboard de Vercel pueda llamar
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoInsight BNB (Cesar)",
    description="Clasificación de zona crítica nocturna con BernoulliNB",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Schemas Pydantic
# ──────────────────────────────────────────────────────────────
class SegmentoInput(BaseModel):
    """Mismos 6 inputs que el K-Means. Las features binarias específicas
    de BernoulliNB (segmento_corto, vel_alta, etc.) se derivan abajo."""
    length:        float = Field(..., ge=5,  le=2500)
    speed_kph:     float = Field(..., ge=20, le=110)
    bearing:       float = Field(..., ge=0,  le=360)
    travel_time:   Optional[float] = Field(None, ge=0, le=500)
    hora_del_dia:  int   = Field(..., ge=0, le=23)
    zona_encoded:  int   = Field(..., ge=0, le=3)
    es_fin_semana: int   = Field(0,   ge=0, le=1)

class PrediccionOutput(BaseModel):
    clase:             str
    probabilidad:      float
    color:             str
    probas_por_clase:  dict
    # Compatibilidad con el shape de K-Means para el formulario del dashboard
    cluster:           int
    nombre:            str
    nivel_alerta:      str
    score_medio:       float
    recomendacion:     str

# ──────────────────────────────────────────────────────────────
# Preprocesamiento: deriva las binarias típicas del proyecto
# ──────────────────────────────────────────────────────────────
def construir_features(seg: SegmentoInput) -> pd.DataFrame:
    """BernoulliNB necesita features binarias. Las derivamos todas y el
    DataFrame final se queda solo con las que están en FEATURES del metadata."""
    travel_time = seg.travel_time
    if travel_time is None:
        travel_time = round(seg.length / (seg.speed_kph * 1000 / 3600), 2)

    es_horario_peligroso = 1 if (seg.hora_del_dia >= 20 or seg.hora_del_dia < 2) else 0
    enc_manana = (60  <= seg.bearing <= 120) and (6  <= seg.hora_del_dia <= 9)
    enc_tarde  = (240 <= seg.bearing <= 300) and (17 <= seg.hora_del_dia <= 20)
    riesgo_encandilamiento = 1 if (enc_manana or enc_tarde) else 0
    encandilamiento_tarde  = 1 if enc_tarde else 0

    segmento_corto = 1 if seg.length < 100 else 0
    segmento_largo = 1 if seg.length > 300 else 0
    vel_alta       = 1 if seg.speed_kph >= 60 else 0
    vel_baja       = 1 if seg.speed_kph <= 40 else 0
    orient_ns      = 1 if (seg.bearing <= 30  or seg.bearing >= 330 or 150 <= seg.bearing <= 210) else 0
    orient_eo      = 1 if (60 <= seg.bearing <= 120 or 240 <= seg.bearing <= 300) else 0

    valores = {
        "segmento_corto":          segmento_corto,
        "segmento_largo":          segmento_largo,
        "vel_alta":                vel_alta,
        "vel_baja":                vel_baja,
        "orient_ns":               orient_ns,
        "orient_eo":               orient_eo,
        "es_horario_peligroso":    es_horario_peligroso,
        "es_fin_semana":           seg.es_fin_semana,
        "riesgo_encandilamiento":  riesgo_encandilamiento,
        "encandilamiento_tarde":   encandilamiento_tarde,
        # No-binarias (por si tu modelo las usa pese a ser BNB)
        "length":        seg.length,
        "speed_kph":     seg.speed_kph,
        "bearing":       seg.bearing,
        "travel_time":   travel_time,
        "hora_del_dia":  seg.hora_del_dia,
        "zona_encoded":  seg.zona_encoded,
    }
    return pd.DataFrame([[valores.get(f, 0) for f in FEATURES]], columns=FEATURES)

# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "servicio": "AutoInsight BNB",
        "modelo":   "BernoulliNB",
        "version":  "1.0.0",
        "accuracy": metadata.get("accuracy"),
        "recall":   metadata.get("recall"),
        "target":   metadata.get("target", "es_zona_critica_nocturna"),
        "estado":   "activo",
        "endpoints": {
            "GET /classes":         "Lista de clases (crítica / no crítica)",
            "GET /probabilities":   "P(feature=1 | clase) por cada feature",
            "GET /metrics":         "Accuracy, recall, precision, F1, matriz",
            "POST /predict":        "Predice clase desde JSON",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": True}

@app.get("/classes")
def classes():
    """Lista de clases con conteo (si el metadata lo trae) y color."""
    counts = metadata.get("counts", {}) or metadata.get("class_counts", {})
    return {
        "classes": [
            {
                "label": str(clase),
                "count": int(counts.get(clase, 0)),
                "color": color_de(clase, i),
            }
            for i, clase in enumerate(CLASSES)
        ],
    }

@app.get("/probabilities")
def probabilities():
    """Matriz de probabilidades condicionales P(feature=1 | clase).
    Es el panel estrella de BernoulliNB: muestra qué tan probable es
    cada feature dado cada clase. Diferencias grandes = la feature
    discrimina bien."""
    # En sklearn, `feature_log_prob_` tiene shape (n_classes, n_features)
    # y contiene log(P(feature=1 | clase)). Lo exponenciamos.
    feature_log_prob = model.feature_log_prob_  # (n_classes, n_features)
    probs = np.exp(feature_log_prob)
    matrix = {}
    for ci, clase in enumerate(CLASSES):
        matrix[str(clase)] = {
            FEATURES[fi]: round(float(probs[ci, fi]), 4)
            for fi in range(len(FEATURES))
        }
    return {
        "classes":  [str(c) for c in CLASSES],
        "features": FEATURES,
        "matrix":   matrix,
    }

@app.get("/metrics")
def metrics():
    """Métricas guardadas en el metadata."""
    return {
        "accuracy":         metadata.get("accuracy"),
        "accuracy_cv":      metadata.get("accuracy_cv"),
        "precision":        metadata.get("precision"),
        "recall":           metadata.get("recall"),
        "f1":               metadata.get("f1"),
        "confusion_matrix": metadata.get("confusion_matrix"),
        "classes":          [str(c) for c in CLASSES],
    }

@app.post("/predict", response_model=PrediccionOutput)
def predict(seg: SegmentoInput):
    """Predice clase (crítica / no crítica) para una calle."""
    try:
        X      = construir_features(seg)
        probas = model.predict_proba(X)[0]
        idx    = int(np.argmax(probas))
        clase  = CLASSES[idx]
        color  = color_de(clase, idx)
        # Mapeo crítica → ROJO / no crítica → VERDE (nivel_alerta del dashboard)
        es_critica = str(clase).lower().startswith("crit") or str(clase) == "1"
        nivel = "ROJO" if es_critica else "VERDE"
        nombre = "Zona Crítica Nocturna (BNB)" if es_critica else "Zona No Crítica (BNB)"
        recomendacion = (
            "Patrullaje intensivo recomendado (BernoulliNB)"
            if es_critica
            else "Riesgo bajo según BernoulliNB"
        )
        return PrediccionOutput(
            clase=str(clase),
            probabilidad=round(float(probas[idx]), 4),
            color=color,
            probas_por_clase={
                str(CLASSES[i]): round(float(p), 4) for i, p in enumerate(probas)
            },
            # Campos extra para que el formulario del dashboard funcione:
            cluster=idx,
            nombre=nombre,
            nivel_alerta=nivel,
            score_medio=round(float(probas[idx]), 4),
            recomendacion=recomendacion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

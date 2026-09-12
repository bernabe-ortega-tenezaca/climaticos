from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="API de Prediccion de Potencial Energetico en Ecuador",
    description="API que predice el potencial solar, eolico e hidrico basado en variables geograficas y climaticas.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    latitude: float = Field(..., ge=-5.0, le=2.0)
    longitude: float = Field(..., ge=-92.0, le=-75.0)
    month: int = Field(..., ge=1, le=12)
    temp_avg: float = Field(..., ge=-10.0, le=40.0)
    wind_speed_avg: float = Field(..., ge=0.0, le=30.0)
    humidity_avg: float = Field(..., ge=0.0, le=100.0)
    # Nota: humidity_avg solo se usa para solar/eolico, NO para hidrico
    # (eliminado del feature set hidrico para evitar data leakage)

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": -0.2295,
                "longitude": -78.5243,
                "month": 6,
                "temp_avg": 15.0,
                "wind_speed_avg": 10.5,
                "humidity_avg": 70.0
            }
        }

def _load_models():
    """Carga modelos en cascada: v4 → v3 → v2 como fallback."""
    for version in ('v5', 'v4', 'v3', 'v2'):
        try:
            solar = joblib.load(MODELS_DIR / f'model_solar_{version}.joblib')
            wind  = joblib.load(MODELS_DIR / f'model_wind_{version}.joblib')
            hydro = joblib.load(MODELS_DIR / f'model_hydro_{version}.joblib')
            desc = "23 provincias sin Galápagos" if version == 'v5' else "PVGIS 24 provincias, sin leakage"
            logger.info("Modelos %s cargados (%s)", version, desc)
            return solar, wind, hydro
        except FileNotFoundError:
            logger.warning("Modelos %s no encontrados, intentando version anterior...", version)
    logger.error("No se encontraron modelos en %s", MODELS_DIR)
    return None, None, None

model_solar, model_wind, model_hydro = _load_models()

@app.get("/health")
def health() -> dict:
    models_ok = all([model_solar is not None, model_wind is not None, model_hydro is not None])
    return {
        "status": "ok" if models_ok else "degraded",
        "models_loaded": models_ok
    }

@app.post("/predict")
def predict(data: PredictionInput) -> dict:
    if not all([model_solar, model_wind, model_hydro]):
        raise HTTPException(status_code=503, detail="Modelos no disponibles")

    # Usa el orden de features de cada modelo para garantizar compatibilidad
    # Solar/Eolico usan humidity_avg; Hidrico NO (evita data leakage)
    full_df = pd.DataFrame([data.model_dump()])

    feature_order_sw = list(model_solar.feature_names_in_)
    input_sw = full_df[feature_order_sw]

    feature_order_h = list(model_hydro.feature_names_in_)
    input_h = full_df[feature_order_h]

    try:
        prediction_solar = float(model_solar.predict(input_sw)[0])
        prediction_wind = float(model_wind.predict(input_sw)[0])
        prediction_hydro = float(model_hydro.predict(input_h)[0])

        return {
            "potencial_solar_kwh_por_m2": round(prediction_solar, 4),
            "potencial_eolico_mwh": round(prediction_wind, 4),
            "potencial_hidrico_mwh": round(prediction_hydro, 4)
        }
    except Exception as e:
        logger.error("Error en prediccion: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")

@app.get("/")
def read_root() -> dict:
    return {
        "message": "Bienvenido a la API de Prediccion de Potencial Energetico de Ecuador. Visita /docs para ver la documentacion interactiva."
    }

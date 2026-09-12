# API REST — FastAPI

**Archivo:** `src/api/main.py`
**Servidor:** `uvicorn src.api.main:app --reload`
**URL local:** `http://127.0.0.1:8000`
**Docs interactivos:** `http://127.0.0.1:8000/docs`

## Endpoints

### `POST /predict`

Predice el potencial energético (solar, eólico, hídrico) para una ubicación y condiciones climáticas dadas.

**Request body (JSON):**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `latitude` | float | Latitud de la ubicación |
| `longitude` | float | Longitud de la ubicación |
| `month` | int | Mes del año (1-12) |
| `temp_avg` | float | Temperatura promedio (°C) |
| `wind_speed_avg` | float | Velocidad del viento promedio (m/s) |
| `precip_total` | float | Precipitación total del mes (mm) |
| `humidity_avg` | float | Humedad relativa promedio (%) |

**Ejemplo:**

```json
{
  "latitude": -0.2295,
  "longitude": -78.5243,
  "month": 6,
  "temp_avg": 15.0,
  "wind_speed_avg": 10.5,
  "precip_total": 80.0,
  "humidity_avg": 70.0
}
```

**Response (JSON):**

```json
{
  "potencial_solar_kwh_per_m2": 18.52,
  "potencial_eolico_mwh": 34.95,
  "potencial_hidrico_mm": 112410.68
}
```

### `GET /`

Endpoint raíz de verificación.

## Modelos cargados

Tres modelos `joblib` entrenados con Random Forest (v1):
- `models/model_solar_v1.joblib`
- `models/model_wind_v1.joblib`
- `models/model_hydro_v1.joblib`

## Flujo interno

1. Valida entrada con Pydantic (`PredictionInput`)
2. Carga los 3 modelos al iniciar (una sola vez)
3. Prepara DataFrame con el orden exacto de features del entrenamiento
4. Ejecuta `model.predict()` para cada tecnología
5. Retorna JSON con los 3 valores redondeados a 4 decimales

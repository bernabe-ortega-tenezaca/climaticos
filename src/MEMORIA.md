# Memoria del Proyecto — Atlas Energético de Ecuador

> **Propósito:** Este documento captura el contexto completo del proyecto para retomar el trabajo sin pérdida de información en futuras sesiones.
> **Publicación IEEE asociada:** Predicción de potencial energético renovable en Ecuador mediante Machine Learning.
> **Repositorio:** `https://github.com/bernabe-ortega-tenezaca/energias_renovables.git`

---

## 1. Identidad del Proyecto

| Atributo | Valor |
|----------|-------|
| Nombre | Atlas Energético de Ecuador |
| Objetivo | Predecir potencial energético renovable (solar, eólico, hídrico) para las 24 provincias de Ecuador |
| Enfoque | Ciencia de datos + Física de ingeniería + Machine Learning |
| Stack | Python 3.10, scikit-learn, FastAPI, Streamlit, pvlib, geopandas |
| Estado funcional | Completo: pipeline ETL → modelos ML → API → Dashboard |
| Problema conocido | Data leakage en target hídrico por coordenadas compartidas entre provincias |

---

## 2. Arquitectura del Sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Notebooks       │ ──▶ │  src/api/        │ ──▶ │  src/app/        │
│  (Pipeline ETL)  │     │  main.py         │     │  streamlit_app.py│
│                  │     │  (FastAPI)       │     │  (Streamlit)     │
│  1. adquisicion  │     │                  │     │                  │
│  2. datos NASA   │     │  POST /predict   │     │  Mapa interactivo│
│  3. processing   │     │  GET /           │     │  Sliders climát. │
│  4. EDA          │     │                  │     │  Reporte PDF     │
│  5. training     │     └──────────────────┘     └──────────────────┘
└─────────────────┘
```

Los notebooks generan `data/processed/final_dataset_ecuador.csv`. Ese dataset entrena los modelos en `7_model_training.ipynb`. Los modelos `.joblib` se guardan en `models/`. La API carga esos modelos y el dashboard los consume.

**Flujo de datos:**
1. Fuentes externas (PVGIS, NASA POWER, Meteostat) → archivos CSV crudos
2. Notebook `5_processing` → dataset final con features + targets físicos
3. Notebook `7_model_training` → modelos .joblib + tabla comparativa
4. `src/api/main.py` → carga modelos, expone endpoint REST
5. `src/app/streamlit_app.py` → UI que consume la API

---

## 3. Dataset

### 3.1 Origen y estructura

| Aspecto | Detalle |
|---------|---------|
| Registros | 288 (24 provincias × 12 meses) |
| Período climático | 2015–2023 (datos diarios promediados a mensuales) |
| Fuente solar | PVGIS (Typical Meteorological Year) |
| Fuente meteorológica | NASA POWER (API REST), fallback Meteostat |
| Features solar/eólico (6) | latitude, longitude, month, temp_avg, wind_speed_avg, humidity_avg |
| Features hídrico (5) | latitude, longitude, month, temp_avg, wind_speed_avg |
| Targets (3) | target_solar_kwh_per_m2, target_wind_mwh, target_hydro_mwh |

### 3.2 Feature engineering — Targets físicos

**Potencial Solar:**
- Irradiación en plano inclinado (POA) usando pvlib con modelo de Perez
- Inclinación del panel = latitud × 1.1, orientación sur (180° azimut)
- Performance ratio del sistema: 15%

**Potencial Eólico:**
- Extrapolación de viento de 2 m → 100 m (altura de turbina)
- Coeficiente de Hellman regionalizado por zona: Costa α=0.14, Sierra α=0.25, Amazonía α=0.20, Insular α=0.10
- Curva de potencia simplificada para turbina de 2 MW

**Potencial Hídrico:**
- Ecuación física: E = ρ · g · Q · H · η · t
- Caudal Q estimado desde precipitación mensual
- Coeficiente de escorrentía variable según humedad y temperatura
- Área de captación: 500 km², altura: 200 m, eficiencia turbina: 85%

### 3.3 Problemas del dataset (corregidos en v3)

~~1. **Data leakage en target hídrico:** Provincias como Bolívar/Chimborazo y Cotopaxi/Tungurahua comparten coordenadas de capital, resultando en features idénticas pero targets con data leakage. El R² de 1.0 en regresión lineal para hídrico lo confirma.~~ ✅ **Corregido:** Se usan centroides geométricos provinciales y se eliminó `humidity_avg` del feature set hídrico.

~~2. **NaN en humedad:** `humidity_avg` tiene valores ausentes para ~8 provincias costeras (Galápagos, Guayas, Manabí, Santa Elena, Santo Domingo, Orellana, Sucumbíos) porque NASA POWER no devuelve RH2M para esas ubicaciones.~~ ✅ **Corregido:** Imputación por zona climática (Costa, Sierra, Amazonía, Insular) con mediana de cada zona.

~~3. **Target hídrico tiene target_hydro_mm** en el EDA (código de EDA) pero **target_hydro_mwh** en el CSV real y en el entrenamiento. Hay inconsistencia de unidades en el notebook de EDA.~~ ✅ **Corregido:** Unidades unificadas a MWh en todo el pipeline.

---

## 4. Modelos de Machine Learning

### 4.1 Algoritmos comparados

| Algoritmo | Hiperparámetros | Librería |
|-----------|----------------|----------|
| Regresión Lineal | default | sklearn |
| Random Forest | n_estimators=100, random_state=42, n_jobs=-1 | sklearn |
| Gradient Boosting | n_estimators=100, random_state=42 | sklearn |

### 4.2 Resultados — Tabla IV (v4 — resultados definitivos)

| Modelo | Algoritmo | MAE | RMSE | R² |
|--------|-----------|-----|------|----|
| Solar | Reg. Lineal | 2.2884 | 2.7920 | 0.3448 |
| Solar | Random Forest | 1.3491 | 1.7780 | 0.7343 |
| Solar | Gradient Boosting | 1.1700 | 1.5156 | **0.8069** |
| Eólico | Reg. Lineal | 22.5250 | 34.9086 | 0.4470 |
| Eólico | Random Forest | 4.6276 | 8.7191 | 0.9655 |
| Eólico | Gradient Boosting | 3.4303 | 6.8332 | **0.9788** |
| Hídrico | Reg. Lineal | 3916.74 | 4852.81 | 0.5170 |
| Hídrico | Random Forest | 1806.31 | 2370.15 | 0.8848 |
| Hídrico | Gradient Boosting | 1640.28 | 2169.84 | **0.9034** |

_Unidades: Solar → kWh/m², Eólico y Hídrico → MWh. Split aleatorio 80/20 sobre pares (provincia, mes). 288 registros totales._

**Validación cruzada GroupKFold(5) — generalización provincial:**

| Modelo | Algoritmo | R² medio | Desv. Est. |
|--------|-----------|:--------:|:----------:|
| Solar | Random Forest | 0.1276 | 0.0948 |
| Eólico | Random Forest | 0.7035 | 0.2186 |
| Eólico | Gradient Boosting | 0.7412 | 0.2114 |
| Hídrico | Random Forest | 0.5194 | 0.1474 |

_Nota: R² CV < R² test es esperado — GroupKFold excluye provincias completas, midiendo extrapolación espacial._

### 4.3 Validación cruzada 10-fold

Validación cruzada con KFold(shuffle=True, random_state=42) implementada en el notebook `7_model_training`. Todos los modelos pasan por 10 particiones con cálculo de R² medio y desviación estándar.

### 4.4 Feature importance

Random Forest y Gradient Boosting permiten identificar las variables más influyentes. En general:
- **Solar:** latitud y mes (estacionalidad) son los predictores más fuertes
- **Eólico:** velocidad del viento y latitud dominan
- **Hídrico:** precipitación total y humedad son las variables clave (con data leakage)

---

## 5. Componentes de Software

### 5.1 API REST (`src/api/main.py`)

| Aspecto | Valor |
|---------|-------|
| Framework | FastAPI |
| Puerto | 8000 |
| Endpoints | `POST /predict`, `GET /` |
| Documentación | `/docs` (Swagger UI) |
| Modelos cargados | 3 (solar, wind, hydro) — Random Forest v1 |
| Validación | Pydantic `PredictionInput` |

**Orden de features (v3):** Solar/Eólico: `['latitude', 'longitude', 'month', 'temp_avg', 'wind_speed_avg', 'humidity_avg']`. Hídrico: `['latitude', 'longitude', 'month', 'temp_avg', 'wind_speed_avg']`.

### 5.2 Dashboard (`src/app/streamlit_app.py`)

| Aspecto | Valor |
|---------|-------|
| Framework | Streamlit |
| Puerto | 8501 (default) |
| Dependencia externa | API en `http://127.0.0.1:8000` |
| Mapa | Folium con capa GeoJSON de provincias |
| Entrada | Click en mapa + 5 sliders climáticos |
| Salida | 3 métricas + PDF descargable |

**Limitación conocida:** El dashboard espera el shapefile `ne_10m_admin_1_states_provinces.shp` en `data/external/`, que está en `.gitignore`. Para ejecutar desde clon fresco, se debe descargar manualmente desde Natural Earth.

### 5.3 Extractores (`src/extraction/`)

- **`meteostatE.py`:** Clase `MeteostatExtractor` con método `extraer_datos_ecuador()`. Descarga datos de estaciones meteorológicas en Ecuador. Actualmente es fallback en el pipeline.
- **`limites.py`:** Carga polígonos provinciales desde GeoJSON, calcula centroides, genera mapa base. Contiene código de diagnóstico y visualización.

---

## 6. Dependencias (`environment.yml`)

**Conda (canal conda-forge):** python=3.10, numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, geopandas, folium, contextily, requests

**Pip:** fastapi, uvicorn[standard], streamlit, streamlit-folium, meteostat, joblib, pvlib, windpowerlib, elevation, tqdm

**Nota:** `windpowerlib` y `elevation` son dependencias declaradas pero no se usan activamente en el código actual.

---

## 7. Estructura del Repositorio

```
Tesis/
├── MEMORIA.md                    # ← Este documento
├── README.md                     # Documentación principal
├── .gitignore                    # Reglas de exclusión
├── environment.yml               # Entorno conda
├── src/
│   ├── api/main.py               # API REST FastAPI
│   ├── app/streamlit_app.py      # Dashboard Streamlit
│   └── extraction/
│       ├── limites.py            # Límites provinciales y centroides
│       └── meteostatE.py         # Extractor Meteostat
├── notebooks/
│   ├── adquisicion.ipynb         # Paso 1: provincias → GeoJSON
│   ├── 4_datos_climaticosNASA.ipynb  # Paso 2: descarga climática
│   ├── 5_processing.ipynb        # Paso 3: targets energéticos
│   ├── 6_EDA.ipynb               # Paso 4: análisis exploratorio
│   ├── 7_model_training.ipynb    # Paso 5: entrenamiento ML
│   └── tabla_iv_resultados.csv   # Resultados comparativos
├── data/                         # Datos (ignorados en git parcialmente)
│   ├── external/                 # Shapefile + GeoJSON
│   ├── raw/                      # GeoJSON de provincias
│   └── processed/                # CSV generados (ignorados)
├── models/                       # Modelos .joblib (ignorados)
└── docs/                         # Imágenes PNG + documentación MD
    ├── api.md                    # Documentación API
    ├── dashboard.md              # Documentación Dashboard
    ├── extraction.md             # Documentación extractores
    └── notebooks.md              # Documentación notebooks
```

**Archivos en tracking git (esenciales, ~150 KB):**
- Código fuente: `.py`, `.ipynb`
- Config: `.gitignore`, `environment.yml`
- Documentación: `README.md`, `MEMORIA.md`, `docs/*.md`
- Resultados pequeños: `notebooks/tabla_iv_resultados.csv`
- Visualizaciones: `docs/*.png`
- GeoJSON pequeños: `data/external/ecuador_provincias_capitales.geojson`, `data/raw/*.geojson`

**Archivos excluidos (regenerables, ~67 MB ahorrados):**
- Shapefile `ne_10m_admin_1_states_provinces.*` (~35 MB)
- Datos procesados `data/processed/*.csv` (~26 MB)
- Modelos `models/*.joblib` (~5.2 MB)
- PDFs `notebooks/*.pdf` (~792 KB)
- `.DS_Store`, `__pycache__/`, `.conda/`, `.vscode/`

---

## 8. Cómo Ejecutar

### 8.1 Reproducir pipeline completo

```bash
conda env create -f environment.yml
conda activate tesis_renovables

# Paso 1-5 (orden secuencial)
jupyter notebook notebooks/adquisicion.ipynb
jupyter notebook notebooks/4_datos_climaticosNASA.ipynb
jupyter notebook notebooks/5_processing.ipynb
jupyter notebook notebooks/6_EDA.ipynb
jupyter notebook notebooks/7_model_training.ipynb
```

### 8.2 Iniciar API + Dashboard

```bash
# Terminal 1
uvicorn src.api.main:app --reload

# Terminal 2
streamlit run src/app/streamlit_app.py
```

### 8.3 Requisito para Dashboard

El shapefile de provincias debe existir en `data/external/`. Como está en `.gitignore`, hay que descargarlo manualmente:
```
https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
```
Descargar y extraer en `data/external/ne_10m_admin_1_states_provinces.*`

---

## 9. Problemas Conocidos y Tareas Pendientes

### 🔴 Críticos (corregidos en v4)
- [x] **Dataset solo tenía 3 provincias**: Meteostat cubría 6 provincias con discrepancias de nombres (sin tildes). Corregido: PVGIS como fuente primaria para las 24 provincias.
- [x] **Data leakage hídrico (R²=1.0)**: Target era NaN → fallback a cero → R²=1.0 trivial. Corregido: precipitación desde NASA POWER API mensual + `runoff_coeff` fijo 0.35.
- [x] **humidity_avg todo NaN**: `relative_humidity` en Meteostat era 0 non-null. Corregido: se usa `relative_humidity` de PVGIS (dato completo para 24 provincias).
- [x] **Extrapolación Hellman incorrecta**: Se usaba altura de referencia 2 m (Meteostat) con datos PVGIS que son a 10 m. Corregido: `height_ref=10`.
- [x] **Mapas EDA sin datos**: Merge shapefile fallaba en 4 provincias por diferencia de tildes. Corregido: normalización Unicode antes del merge.

### 🟡 Mejoras (corregidas en v3, vigentes)
- [x] **Zonas climáticas**: nombres con tildes corregidos para coincidir con PVGIS.
- [x] **Inconsistencia de unidades en EDA**: Unificado a MWh.
- [x] **Hellman alpha regionalizado**: Costa 0.14, Sierra 0.25, Amazonía 0.20, Insular 0.10.
- [x] **GridSearchCV**: Búsqueda de hiperparámetros con GroupKFold 3-fold.
- [x] **Feature sets separados**: Solar/eólico con 6 features, hídrico con 5 (sin humidity_avg).

### 🟢 Baja prioridad (pendientes)
- [ ] **API actualizada**: Cargar modelos v4 (actualmente carga v3/v1).
- [ ] **Dashboard**: Verificar compatibilidad con feature sets v4.
- [ ] **Refactorizar notebooks a módulos .py**.
- [ ] **CI/CD** con GitHub Actions para validar notebooks.

---

## 10. Historial de Commits

```
0fb00f5 chore: eliminar directorio obsoleto notebooks/4_datos_climaticosNO
0ba55d0 refactor: humanizar comentarios, limpiar gitignore y documentar proyecto completo
2cf0f1c feat: agregada validacion cruzada 10-fold a modelos
9e14716 fix: corregido data leakage en target hidrico, agregados LR y GB
ac4f1da fix: corregido data leakage en target hidrico, agregados LR y GB
73192b9 fix: gitignore
4656f90 backup: proyecto tesis antes de correcciones articulo
e333e90 final
474ef3a final v1
8fb05e0 upd environment.yml
b830c9f Poligonos geoson Ecuador
b6d9e8c GeoJSON
6b4afbd feat. extraer datos Ecuador
ee35b71 Estructura
```

---

## 11. Guía para Próximas Sesiones

Al retomar el trabajo:
1. Leer este `MEMORIA.md` completo
2. `git pull` para sincronizar
3. `conda activate tesis_renovables` para activar entorno
4. Verificar `git status` para ver cambios no commiteados
5. Revisar la sección **Problemas Conocidos** para priorizar tareas
6. Leer `.opencode/AGENTS.md` para instrucciones específicas del asistente IA

---

*Documento actualizado el 27 de mayo de 2026 — versión v4 con pipeline corregido (288 registros, 24 provincias, sin data leakage).*

> **Archivo complementario:** `.opencode/AGENTS.md` contiene instrucciones condensadas para asistentes IA que retomen el proyecto.

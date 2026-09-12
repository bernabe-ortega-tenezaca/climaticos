# Atlas Energético de Ecuador

> **Proyecto de tesis de grado — Publicación IEEE**
> Predicción de potencial energético renovable (solar, eólico e hídrico) en las 24 provincias de Ecuador mediante Machine Learning.

**Nota:** El documento de memoria del proyecto contiene contexto completo, decisiones arquitectónicas y tareas pendientes. Está disponible localmente para los autores.

---

## Estructura del Proyecto

```
Tesis/
├── src/
│   ├── api/main.py               # API REST (FastAPI) para predicciones
│   ├── app/streamlit_app.py      # Dashboard interactivo (Streamlit)
│   └── extraction/
│       ├── limites.py            # Carga, centroides y mapa de provincias
│       └── meteostatE.py         # Clase extractora de datos Meteorológicos
├── notebooks/
│   ├── adquisicion.ipynb         # Paso 1: provincias → GeoJSON
│   ├── 4_datos_climaticosNASA.ipynb  # Paso 2: descarga de datos climáticos
│   ├── 5_processing.ipynb        # Paso 3: cálculo de targets energéticos
│   ├── 6_EDA.ipynb               # Paso 4: análisis exploratorio y mapas
│   └── 7_model_training.ipynb    # Paso 5: entrenamiento y evaluación de modelos
├── data/
│   ├── external/                 # Shapefile mundial + GeoJSON capitales
│   ├── raw/                      # GeoJSON de provincias (polígonos)
│   └── processed/                # Datasets generados (regenerables)
├── models/                       # Modelos .joblib entrenados (regenerables)
├── docs/                         # Imágenes PNG + documentación MD
├── MEMORIA.md                    # Memoria completa del proyecto
├── environment.yml               # Dependencias conda
└── .gitignore
```

---

## Pipeline completo

| Paso | Componente | Descripción |
|------|-----------|-------------|
| 1 | `adquisicion.ipynb` | Define las 24 provincias con centroides geométricos (antes capitales) y exporta a GeoJSON |
| 2 | `4_datos_climaticosNASA.ipynb` | Descarga datos climáticos diarios (2015-2023) desde PVGIS + NASA POWER (fallback Meteostat) |
| 3 | `5_processing.ipynb` | Calcula targets físicos de potencial energético usando pvlib y fórmulas de ingeniería |
| 4 | `6_EDA.ipynb` | Análisis exploratorio: gráficos, mapas coropléticos, matriz de correlación |
| 5 | `7_model_training.ipynb` | Entrena 3 algoritmos (LR, RF, GB) × 3 energías = 9 modelos. Validación cruzada 10-fold. GridSearchCV para optimización de hiperparámetros |
| 6 | `src/api/main.py` | API REST que carga los 3 modelos y expone `POST /predict` |
| 7 | `src/app/streamlit_app.py` | Dashboard con mapa interactivo, sliders climáticos y generación de PDF |

---

## Dataset

- **288 registros** (24 provincias × 12 meses)
- **6 features solar/eólico**: latitud, longitud, mes, temperatura, viento, humedad
- **5 features hídrico**: latitud, longitud, mes, temperatura, viento (sin humedad para evitar data leakage)
- **3 targets**: potencial solar (kWh/m²), eólico (MWh), hídrico (MWh)

## Modelos y resultados (v4)

| Energía | Reg. Lineal (R²) | Random Forest (R²) | Gradient Boosting (R²) |
|---------|:----------------:|:------------------:|:----------------------:|
| Solar   | 0.34 | 0.73 | **0.81** |
| Eólico  | 0.45 | 0.97 | **0.98** |
| Hídrico | 0.52 | 0.88 | **0.90** |

_Validación cruzada GroupKFold(5) — generalización provincial: Eólico RF R²=0.70, Hídrico RF R²=0.52, Solar RF R²=0.13._

**Correcciones v4 (sobre v3):**
1. **Fuente de datos:** PVGIS como fuente primaria para las 24 provincias (Meteostat solo cubría 3)
2. **Precipitación hídrica:** NASA POWER API mensual (independiente de los features → sin leakage)
3. **Leakage hídrico eliminado:** `runoff_coeff` fijo 0.35 (antes dependía de `humidity_avg`/`temp_avg`)
4. **Viento corregido:** Hellman 10 m→100 m (PVGIS da viento a 10 m, no 2 m)
5. **Zonas climáticas:** nombres con tildes corregidos para coincidir con nomenclatura PVGIS
6. **Mapas EDA:** merge shapefile con normalización Unicode (antes 4 provincias sin match)
5. Hiperparámetros: optimizados con GridSearchCV + GroupKFold

---

## Cómo ejecutar

### Entorno conda

```bash
conda env create -f environment.yml
conda activate tesis_renovables
```

### Reproducir pipeline

```bash
jupyter notebook notebooks/adquisicion.ipynb
jupyter notebook notebooks/4_datos_climaticosNASA.ipynb
jupyter notebook notebooks/5_processing.ipynb
jupyter notebook notebooks/6_EDA.ipynb
jupyter notebook notebooks/7_model_training.ipynb
```

### Iniciar API + Dashboard

```bash
# Terminal 1: API REST
uvicorn src.api.main:app --reload

# Terminal 2: Dashboard
streamlit run src/app/streamlit_app.py
```

---

## Documentación detallada

| Documento | Contenido |
|-----------|-----------|
| [`MEMORIA.md`](MEMORIA.md) | Contexto completo del proyecto, decisiones, problemas conocidos y tareas |
| [`docs/api.md`](docs/api.md) | Documentación de la API REST (endpoints, request/response) |
| [`docs/dashboard.md`](docs/dashboard.md) | Documentación del Dashboard Streamlit |
| [`docs/extraction.md`](docs/extraction.md) | Documentación de extractores de datos |
| [`docs/notebooks.md`](docs/notebooks.md) | Documentación del pipeline de notebooks |

---

## Notas

- Los archivos en `data/processed/`, `models/`, y `notebooks/*.pdf` se regeneran ejecutando los notebooks y están excluidos de git.
- El shapefile `ne_10m_admin_1_states_provinces.*` (35 MB) se ignora. Descargar de Natural Earth si se necesita el dashboard.
- Ver `MEMORIA.md` sección 9 para lista completa de problemas conocidos y mejoras pendientes.

**Repositorio:** `https://github.com/bernabe-ortega-tenezaca/energias_renovables.git`

# Notebooks — Pipeline de datos

Los notebooks deben ejecutarse en orden secuencial, ya que cada uno depende de los outputs del anterior.

---

## 1. `adquisicion.ipynb` — Adquisición de provincias

**Salida:** `data/external/ecuador_provincias_capitales.geojson`

Define manualmente las 24 provincias de Ecuador con sus capitales y coordenadas geográficas. Convierte el DataFrame a GeoDataFrame con CRS EPSG:4326 y exporta a GeoJSON.

**Estructura creada:**
- `provincia`: nombre de la provincia
- `capital`: ciudad capital
- `latitude`, `longitude`: coordenadas decimales
- `geometry`: punto geográfico (shapely Point)

---

## 2. `4_datos_climaticosNASA.ipynb` — Descarga climática

**Salidas:**
- `data/processed/pvgis_data_all_provinces.csv` — datos solares PVGIS (año típico)
- `data/processed/meteostat_data_all_provinces.csv` — datos meteorológicos históricos

Descarga datos climáticos diarios para cada provincia (2015-2023):

| Fuente | Datos | Uso |
|--------|-------|-----|
| **PVGIS** (pvlib) | GHI, DNI, DHI, temperatura | Cálculo de potencial solar |
| **NASA POWER** (API REST) | T2M, WS2M, PRECTOTCORR, RH2M | Features climáticas |
| **Meteostat** (fallback) | Datos meteorológicos estándar | Backup si NASA falla |

**Estrategia híbrida:** Primero intenta Meteostat; si falla o devuelve vacío, usa NASA POWER. Todos los DataFrames se estandarizan al mismo formato de columnas.

---

## 3. `5_processing.ipynb` — Cálculo de targets energéticos

**Salida:** `data/processed/final_dataset_ecuador.csv`

Construye el dataset final con 7 features + 3 targets por provincia y mes.

### Cálculo de targets

#### Potencial Solar (`target_solar_kwh_per_m2`)
- Usa pvlib para calcular irradiación en plano inclinado (POA)
- Modelo de Perez para radiación difusa
- Inclinación del panel = latitud × 1.1
- Performance ratio del sistema: 15%

#### Potencial Eólico (`target_wind_mwh`)
- Extrapolación de velocidad del viento de 2m a 100m (altura de turbina)
- Coeficiente de Hellman α = 0.14
- Curva de potencia simple para turbina de 2 MW

#### Potencial Hídrico (`target_hydro_mwh`)
- Estimación física: E = ρ · g · Q · H · η · t
- Caudal Q estimado desde precipitación con coeficiente de escorrentía variable
- Coeficiente de escorrentía ajustado por humedad y temperatura
- Área de captación: 500 km², altura: 200 m, eficiencia: 85%

---

## 4. `6_EDA.ipynb` — Análisis exploratorio

**Salidas:** 8 imágenes en `docs/`

### Análisis incluidos
1. **Vista general**: primeras filas, info(), describe()
2. **Barras por provincia**: potencial solar, eólico e hídrico promedio mensual
3. **Estacionalidad**: promedio nacional por mes para las 3 energías
4. **Matriz de correlación**: mapa de calor entre todas las variables numéricas
5. **Mapas coropléticos**: 3 mapas (solar, eólico, hídrico) usando shapefile de provincias
6. **Diagnóstico de shapefile**: verificación de columnas disponibles

---

## 5. `7_model_training.ipynb` — Entrenamiento de modelos

**Salidas:**
- `models/model_solar_v1.joblib`
- `models/model_wind_v1.joblib`
- `models/model_hydro_v1.joblib`
- `notebooks/tabla_iv_resultados.csv` (tabla comparativa)

### Algoritmos entrenados
| Algoritmo | Librería | Hiperparámetros |
|-----------|----------|-----------------|
| Regresión Lineal | sklearn | default |
| Random Forest | sklearn | n_estimators=100, random_state=42 |
| Gradient Boosting | sklearn | n_estimators=100, random_state=42 |

### Evaluación
- Train/test split: 80/20
- Métricas: MAE, RMSE, R²
- Validación cruzada: 10-fold con shuffle
- Feature importance para modelos basados en árboles

### Diagnóstico de colinealidad
Incluye un bloque de diagnóstico para detectar data leakage en el modelo hídrico, analizando correlación features-target y coeficientes de regresión lineal.

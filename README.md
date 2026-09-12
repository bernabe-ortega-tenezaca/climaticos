# Potencial renovable del Ecuador — banco de pruebas de generalización espacial

Código, datos y modelos del artículo **«Límites espaciales del aprendizaje
automático en energías renovables, Ecuador»** (*Spatial limits of machine
learning for renewable energy potential in Ecuador*), enviado a la revista
**INGENIUS** (Universidad Politécnica Salesiana del Ecuador).

El trabajo evalúa modelos de potencial solar, eólico e hidroeléctrico para las
24 provincias del Ecuador bajo **dos protocolos de validación contrastados**:
una partición aleatoria 80/20, que actúa como techo optimista, y una validación
cruzada `GroupKFold(5)` que retiene provincias completas y actúa como piso
operativo. La brecha entre ambos es el resultado central.

---

## Versión que reproduce el artículo: **v6**

El repositorio conserva el historial de versiones del modelo (v1–v6). **Las
cifras publicadas corresponden únicamente a la v6**, entrenada el
2026-05-27. Las versiones anteriores se documentan en
[`models/CHANGELOG.md`](models/CHANGELOG.md) y sus metadatos se conservan, pero
sus métricas **no** son las del artículo.

| Tabla del artículo | Archivo |
|---|---|
| Tabla 1 — hiperparámetros | [`models/metadata_v6.json`](models/metadata_v6.json) → `hiperparametros` |
| Tabla 2 — partición aleatoria 80/20 + IC95 bootstrap | [`data/processed/tabla_iv_resultados_v6.csv`](data/processed/tabla_iv_resultados_v6.csv) y [`tabla_bootstrap_ci_v6.csv`](data/processed/tabla_bootstrap_ci_v6.csv) |
| Tabla 3 — GroupKFold(5) y líneas base | [`data/processed/tabla_groupkfold_v6.csv`](data/processed/tabla_groupkfold_v6.csv) y [`tabla_baselines.csv`](data/processed/tabla_baselines.csv) |
| Tabla 4 — fuera de dominio (Galápagos) | [`data/processed/galapagos_ood_results_v6.csv`](data/processed/galapagos_ood_results_v6.csv) |
| Figura 1 — residuos por provincia | [`docs/residuos_solar.png`](docs/), `residuos_eolico.png`, `residuos_hidrico.png` — regenerados con [`scripts/regenerar_mapas.py`](scripts/regenerar_mapas.py) |
| Figura 2 — importancia SHAP | [`docs/shap_solar_v6.png`](docs/), `shap_eolico.png`, `shap_hidrico.png` |

---

## Estructura

```
data/
  raw/          GeoJSON de provincias y caché de respuestas de NASA POWER
  processed/    dataset final, descargas PVGIS y Meteostat, tablas de resultados
  external/     (ver nota sobre Natural Earth, más abajo)
models/         modelos v6 entrenados (.joblib) y metadatos de todas las versiones
notebooks/      flujo completo, en orden de ejecución
src/            código reutilizable: extracción, preprocesamiento, API y app
docs/           figuras generadas y documentación
```

### Notebooks, en orden

| Notebook | Qué hace |
|---|---|
| `adquisicion.ipynb` | Límites provinciales y centroides de capitales |
| `4_datos_climaticosNASA.ipynb` | Descarga de NASA POWER (2015–2023) con caché local |
| `5_processing.ipynb` | Cálculo de las variables objetivo y ensamblado del dataset |
| `6_EDA.ipynb` | Análisis exploratorio y mapas |
| `7_model_training.ipynb` | **Entrenamiento v6, GroupKFold, bootstrap, SHAP y sonda Galápagos** |

El último es el que produce todas las cifras del artículo.

---

## Reproducción

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/7_model_training.ipynb
```

Semilla fija `random_state = 42` en particiones, modelos y bootstrap.

### Mapas de residuos

Los mapas de `docs/residuos_*.png` se regeneran con:

```bash
python3 scripts/regenerar_mapas.py .
```

El script rehace el dataset v6, reproduce las predicciones fuera de pliegue con
`GroupKFold(5)` sobre los modelos RF ya entrenados y **valida el error de cada
provincia contra las cifras publicadas** antes de dibujar; si no coinciden,
aborta sin generar nada.

Sustituye a la versión anterior de los mapas, que tomaba la geometría de
`data/raw/ecuador_provincias_poligonos.geojson`, cuyos polígonos son cajas
envolventes de cinco vértices y no el contorno real de cada provincia. La
geometría correcta procede del shapefile de Natural Earth.

### Datos de Natural Earth

Los `shapefile` de Natural Earth (34 MB) **no se versionan** para no inflar el
repositorio. Descárgalos y colócalos en `data/external/`:

```bash
curl -L -o /tmp/ne.zip \
  https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip
unzip /tmp/ne.zip -d data/external/
```

Son de dominio público (Natural Earth, escala 1:10 m, WGS 84 / EPSG:4326).

---

## Metodología, en resumen

- **Datos**: PVGIS-TMY (irradiancia), NASA POWER (reanálisis diario 2015–2023,
  agregado a climatología mensual) y estaciones Meteostat en las siete
  provincias donde hay registros fiables.
- **Objetivos**, calculados por simulación física porque no existen registros
  medidos de generación a escala provincial: modelo de transposición de Perez
  con η = 0,15 (solar); ley de Hellman 10→100 m más curva de potencia de un
  aerogenerador de 2 MW (eólico); central a filo de agua con coeficiente de
  escorrentía 0,35 (hidroeléctrico).
- **Predictores**: latitud, longitud, mes, temperatura, velocidad del viento,
  humedad y elevación. El modelo hidroeléctrico excluye humedad y precipitación.
- **Modelos**: regresión lineal, bosques aleatorios y potenciación del
  gradiente (`scikit-learn`), con `GridSearchCV` y bucle interno
  `GroupKFold(3)` estratificado por provincia.
- **Validación**: partición 80/20 y `GroupKFold(5)` por provincia, contra dos
  líneas base triviales (media global y media climatológica mensual), con
  intervalos bootstrap de 1000 remuestreos y Galápagos retenido como sonda
  fuera de dominio.

---

## Cita

El artículo está en evaluación. Hasta su publicación, cita este repositorio.

## Licencia

Código bajo licencia MIT. Los datos derivados conservan las condiciones de sus
fuentes originales (PVGIS, NASA POWER, Meteostat, Natural Earth).

# Potencial renovable del Ecuador — banco de pruebas de generalización espacial

Código, datos y modelos del artículo **«Aprendizaje automático y potencial
renovable en Ecuador: validación espacial»** (*Machine learning and renewable
potential in Ecuador: a spatial validation*), preparado para la revista
**INGENIUS** (Universidad Politécnica Salesiana del Ecuador).

El trabajo compara **dos protocolos de validación** para modelos de tres
indicadores mensuales de potencial (solar, eólico e hidroeléctrico) en las 24
provincias del Ecuador: una partición aleatoria 80/20 y una validación cruzada
que retiene provincias completas. La diferencia entre ambos, y la prueba que
explica su origen, son el resultado central.

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
| Tabla 4 — comprobaciones de robustez | [`scripts/robustez.py`](scripts/robustez.py) → [`data/processed/robustez_resultados.json`](data/processed/robustez_resultados.json) |
| Tabla 5 — fuera de dominio (Galápagos) | [`data/processed/galapagos_ood_results_v6.csv`](data/processed/galapagos_ood_results_v6.csv) |
| Figura 1 — residuos por provincia | [`docs/residuos_solar.png`](docs/), `residuos_eolico.png`, `residuos_hidrico.png` — regenerados con [`scripts/regenerar_mapas.py`](scripts/regenerar_mapas.py) |
| Figura 2 — importancia SHAP | [`docs/shap_solar_v6.png`](docs/), `shap_eolico.png`, `shap_hidrico.png` |

---

## Estructura

```
data/
  raw/          GeoJSON de provincias y caché de respuestas de NASA POWER
  processed/    dataset final, descarga de PVGIS, tablas de resultados
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

### Comprobaciones de robustez

```bash
python3 scripts/robustez.py .            # ~10 min
python3 scripts/referencia_precisa.py .
```

`robustez.py` no reentrena los modelos publicados. Reproduce primero la Tabla 3
como control y después calcula: (1) la validación espacial anidada; (2) la
validación que deja fuera una provincia cada vez, con intervalo por remuestreo
de provincias y prueba de Wilcoxon pareada contra la línea base mensual; (3)
200 asignaciones aleatorias de provincias a cinco pliegues; y (4) un bosque
aleatorio que solo recibe latitud, longitud, elevación y mes.

`referencia_precisa.py` recalcula las Tablas 2, 3 y 5 sin redondear. Los CSV
guardan cuatro decimales, y volver a redondearlos a tres puede desplazar el
último dígito; el artículo usa los valores de este script.

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

- **Datos**: para las coordenadas de cada capital provincial, un año
  meteorológico típico horario de PVGIS (irradiancia global, directa y difusa,
  temperatura, humedad relativa y viento a 10 m) y la precipitación mensual
  media 2015–2023 de NASA POWER. Todo procede de productos satelitales o de
  reanálisis; no intervienen mediciones de superficie. El archivo
  `meteostat_data_all_provinces.csv` se conserva de versiones anteriores y
  **no se usa** en la v6.
- **Indicadores** (variables objetivo): energía mensual de una instalación de
  referencia idéntica en todos los sitios, porque no hay registros de
  generación con cobertura provincial. Solar: transposición isotrópica de
  `pvlib` a un plano inclinado 0,9·|latitud| (mínimo 5°), con eficiencia 0,15.
  Eólico: ley potencial de 10 a 100 m con α = 0,14 (Costa), 0,25 (Sierra),
  0,20 (Amazonía) y 0,10 (Galápagos), y curva de un aerogenerador de 2 MW.
  Hidroeléctrico: central de pasada con cuenca de 500 km², escorrentía 0,35,
  salto de 200 m y eficiencia 0,85, es decir, 81,07 MWh por milímetro mensual
  de precipitación. No son estimaciones del potencial técnico de cada
  provincia.
- **Predictores**: latitud, longitud, mes, elevación y las medias mensuales de
  temperatura, velocidad del viento y humedad relativa. Se excluyen la
  irradiancia y la precipitación, que determinan los indicadores solar e
  hidroeléctrico; el modelo hidroeléctrico prescinde además de la humedad. La
  velocidad del viento sí es predictor del indicador eólico, que se calcula
  con la misma serie: ese modelo es una función de transferencia.
- **Modelos**: regresión lineal, bosques aleatorios y potenciación del
  gradiente (`scikit-learn`), con `GridSearchCV` y bucle interno
  `GroupKFold(3)` agrupado por provincia.
- **Validación**: partición 80/20 y `GroupKFold(5)` por provincia, contra dos
  líneas base triviales (media global y media climatológica mensual), con
  intervalos bootstrap de 1000 remuestreos y Galápagos retenido como sonda
  fuera de dominio.

---

## Cita

El artículo está en evaluación. Hasta su publicación, cita este repositorio.

## Licencia

Código bajo licencia MIT. Los datos derivados conservan las condiciones de sus
fuentes originales (PVGIS, NASA POWER, Natural Earth).

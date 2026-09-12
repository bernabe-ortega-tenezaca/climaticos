# Changelog de Modelos — Atlas Energético de Ecuador

## v4 (actual) — Mayo 2026
- **Fuente de datos:** PVGIS como fuente primaria para las 24 provincias (antes Meteostat → solo 3 provincias)
- **Precipitación:** NASA POWER API mensual (PRECTOTCORR, promedio 2015-2023), cacheada en `data/raw/nasa_cache/monthly_prcp.json`
- **Dataset:** 288 registros (24 provincias × 12 meses), sin NaN
- **Data leakage hídrico eliminado:** `runoff_coeff` fijo en 0.35 (antes dependía de `humidity_avg` y `temp_avg` que son features)
- **Viento corregido:** extrapolación Hellman 10 m→100 m (PVGIS entrega viento a 10 m; antes se usaba 2 m→100 m de Meteostat)
- **Zonas climáticas:** nombres corregidos con tildes para coincidir con nombres PVGIS
- **Split:** aleatorio 80/20 sobre pares (provincia, mes)
- **Resultados (test set):** Solar GB R²=0.81, Eólico GB R²=0.98, Hídrico GB R²=0.90
- **CV GroupKFold(5):** Eólico RF R²=0.70, Hídrico RF R²=0.52, Solar RF R²=0.13

## v3 — Mayo 2026
- **Feature set solar/eólico:** `latitude`, `longitude`, `month`, `temp_avg`, `wind_speed_avg`, `humidity_avg`
- **Feature set hídrico:** `latitude`, `longitude`, `month`, `temp_avg`, `wind_speed_avg` (sin `humidity_avg`)
- **Coordenadas:** centroides geométricos provinciales (antes capitales)
- **Imputación:** por zona climática (Costa, Sierra, Amazonía, Insular)
- **Hiperparámetros:** optimizados con GridSearchCV + GroupKFold(5)
- **Hellman alpha:** regionalizado por zona (0.14 Costa, 0.25 Sierra, 0.20 Amazonía, 0.10 Insular)

## v2 — Mayo 2026
- Se eliminó `precip_total` del feature set
- Se agregó imputación por grupo (provincia)
- Se refactorizaron targets a módulo independiente `src/preprocessing/targets.py`

## v1 — Original
- Versión inicial con data leakage (target hídrico R²=1.0 en Regresión Lineal)
- Coordenadas de capitales provinciales
- Imputación con mediana global
- Hellman alpha fijo α=0.14

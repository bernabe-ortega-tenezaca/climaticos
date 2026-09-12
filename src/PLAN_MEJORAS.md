# Plan de Mejoras — Atlas Energético de Ecuador

> Estado actualizado al 23 de mayo de 2026.
> ✅ = Completada | 🔴 = Crítica | 🟡 = Importante | 🟢 = Media | ⏭️ = Saltada

---

## 🔴 Críticas (impactan resultados científicos)

| # | Mejora | Estado | Archivos afectados |
|---|--------|--------|-------------------|
| 1 | **Corregir data leakage en target hídrico** — `precip_total` era feature y también se usaba para calcular `target_hydro_mwh`. Se eliminó `precip_total` del feature set. | ✅ Completada | `5_processing.ipynb` |
| 2 | **Imputar NaN en humidity_avg** — ~8 provincias costeras sin datos de NASA POWER. Se agrega imputación por grupo (provincia) con fallback a media global. | ✅ Completada | `5_processing.ipynb` |
| 3 | **Unificar unidades del target hídrico** — EDA usa `target_hydro_mm`, dataset real tiene `target_hydro_mwh` | ✅ Completada | `6_EDA.ipynb` |

## 🟡 Importantes (calidad del código y reproducibilidad)

| # | Mejora | Estado | Archivos afectados |
|---|--------|--------|-------------------|
| 4 | **Refactorizar targets a módulo independiente** — extraer lógica de targets físicos a `src/preprocessing/targets.py`. | ✅ Completada | `src/preprocessing/targets.py` |
| 5 | **Poblar directorios src/ vacíos** con `.gitkeep` | ✅ Completada | `src/analysis/`, `src/models/`, etc. |
| 6 | **Reemplazar rutas relativas por pathlib centralizado** | ✅ Completada | `src/api/main.py`, `src/app/streamlit_app.py`, `src/extraction/limites.py` |
| 7 | **Reemplazar `print()` por logging estructurado** | ✅ Completada | `src/api/main.py`, `src/extraction/limites.py`, `src/app/streamlit_app.py` |
| 8 | **Limpiar dependencias no utilizadas** — `windpowerlib` y `elevation` | ✅ Completada | `environment.yml` |

## 🟢 Medias (experiencia de desarrollo y despliegue)

| # | Mejora | Estado | Archivos afectados |
|---|--------|--------|-------------------|
| 9 | **Versionar modelos con metadatos** — fecha, R², features, hiperparámetros en JSON | ✅ Completada | `7_model_training.ipynb` |
| 10 | **Agregar tests unitarios** (11 tests, todos pasan) | ✅ Completada | `tests/test_targets.py` |
| 11 | **Separar celdas de diagnóstico** — bloques marcados en notebooks | ✅ Completada | `5_processing.ipynb`, `6_EDA.ipynb` |
| 12 | **Dockerizar API + Dashboard** con docker-compose | ✅ Completada | `Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml` |
| 13 | **Agregar CI/CD con GitHub Actions** (test runner) | ✅ Completada | `.github/workflows/test.yml` |
| 14 | **Cachear descargas NASA POWER** con archivos JSON locales | ✅ Completada | `4_datos_climaticosNASA.ipynb` |

## 🔵 Bajas (cosméticas y documentación)

| # | Mejora | Estado | Archivos afectados |
|---|--------|--------|-------------------|
| 15 | Eliminar notebooks duplicados | ✅ Completada | `tests/borrar.ipynb` eliminado |
| 16 | Agregar docstrings a funciones en `src/` | ✅ Completada | `main.py`, `streamlit_app.py`, `meteostatE.py` |
| 17 | **Estandarizar nombres de variables a español** | ⏭️ Saltada — rompe API contract y orden de features de los modelos entrenados | — |
| 18 | Agregar type hints a funciones | ✅ Completada | `main.py`, `streamlit_app.py`, `meteostatE.py` |
| 19 | Eliminar `.DS_Store` residual | ✅ Completada | `data/`, `notebooks/` |

## Nueva iteración v3 (Mayo 2026)

| # | Mejora | Estado | Archivos afectados |
|---|--------|--------|-------------------|
| 20 | **Feature sets separados por modelo** — Hídrico sin `humidity_avg` para eliminar data leakage | ✅ Completada | `7_model_training.ipynb`, `src/api/main.py` |
| 21 | **Centroides geométricos** — en vez de coordenadas de capitales provinciales | ✅ Completada | `adquisicion.ipynb` |
| 22 | **Imputación por zona climática** — Costa, Sierra, Amazonía, Insular en vez de mediana global | ✅ Completada | `5_processing.ipynb` |
| 23 | **Unificar preprocesamiento** — Random Forest ahora usa `X_train_imp` (imputado) como LR y GB | ✅ Completada | `7_model_training.ipynb` |
| 24 | **GridSearchCV con GroupKFold** — Búsqueda de hiperparámetros para RF y GB | ✅ Completada | `7_model_training.ipynb` |
| 25 | **Hellman alpha regionalizado** — Costa 0.14, Sierra 0.25, Amazonía 0.20, Insular 0.10 | ✅ Completada | `src/preprocessing/targets.py`, `5_processing.ipynb` |
| 26 | **Unidades EDA corregidas** — "mm de precipitación" → "MWh" en títulos y leyendas | ✅ Completada | `6_EDA.ipynb` |
| 27 | **CHANGELOG de modelos** — `models/CHANGELOG.md` con historial v1→v3 | ✅ Completada | `models/CHANGELOG.md` |
| 28 | **Modelos v3** — Exportación como `model_*_v3.joblib` con metadatos actualizados | ✅ Completada | `7_model_training.ipynb`, `models/metadata_v3.json` |
| 29 | **API actualizada** — Carga modelos v3 con fallback a v2; feature sets separados por modelo | ✅ Completada | `src/api/main.py` |

---

**Resumen:** ✅ 23/24 mejoras, ⏭️ 1 saltada. Pipeline completo listo para re-ejecución v3.

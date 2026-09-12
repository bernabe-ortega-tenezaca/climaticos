# Extracción de Datos

## `meteostatE.py` — Extractor Meteostat

**Archivo:** `src/extraction/meteostatE.py`

Clase `MeteostatExtractor` que obtiene datos climáticos históricos desde la API de Meteostat para Ecuador.

### Métodos

#### `extraer_datos_ecuador(inicio, fin)`
- Busca todas las estaciones meteorológicas en Ecuador (`region('EC')`)
- Descarga datos diarios para cada estación en el rango de fechas
- Concatena todos los DataFrames y guarda en `data/raw/meteostat_ecuador.csv`

**Parámetros:**
- `inicio`: `datetime`, inicio del período (default: 2020-01-01)
- `fin`: `datetime`, fin del período (default: 2024-12-31)

**Nota:** Este extractor se usó como fuente primaria en versiones iniciales. En la versión actual del pipeline, Meteostat se usa como **fallback** cuando NASA POWER no está disponible (ver `4_datos_climaticosNASA.ipynb`).

---

## `limites.py` — Límites provinciales

**Archivo:** `src/extraction/limites.py`

Script para cargar, transformar y visualizar los límites provinciales de Ecuador.

### Funcionalidad
- Carga `data/raw/ecuador_provincias_poligonos.geojson`
- Asegura CRS en EPSG:4326 (WGS84)
- Calcula centroides reales de cada provincia (no media del bbox)
- Genera mapa de provincias con códigos INEC y lo guarda en `docs/ecuador_mapa.png`

### Columnas generadas
- `centroid`: geometría del centroide
- `lat`: latitud del centroide
- `lon`: longitud del centroide

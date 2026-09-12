from pathlib import Path
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
DOCS_DIR = BASE_DIR / 'docs'

provincias = gpd.read_file(DATA_DIR / 'raw' / 'ecuador_provincias_poligonos.geojson')

# Verifica y normaliza el sistema de referencia espacial para compatibilidad con las APIs climáticas
logger.info("CRS original: %s", provincias.crs)
if provincias.crs != "EPSG:4326":
    provincias = provincias.to_crs("EPSG:4326")
logger.info("CRS actual: %s", provincias.crs)

# Calcula centroides precisos a partir de la geometría real de cada polígono
provincias["centroid"] = provincias.geometry.centroid
provincias["lat"] = provincias.centroid.y
provincias["lon"] = provincias.centroid.x

fig, ax = plt.subplots(figsize=(10, 8))
provincias.boundary.plot(ax=ax, linewidth=0.8, color='black')
provincias.plot(ax=ax, cmap='tab20', alpha=0.7)
for _, row in provincias.iterrows():
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(x, y, row["codigo"], fontsize=8, ha='center', va='center')
ax.set_title("Provincias de Ecuador (códigos INEC)", fontsize=14)
plt.tight_layout()
plt.savefig(DOCS_DIR / "ecuador_mapa.png", dpi=150)

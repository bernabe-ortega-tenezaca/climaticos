#!/usr/bin/env python3
# =====================================================================
# regenerar_mapas.py
#
# PROBLEMA QUE RESUELVE
# Los mapas de residuos originales (docs/residuos_*.png) dibujan cada
# provincia como un RECTANGULO. La causa no esta en el codigo de
# graficado del notebook 7_model_training.ipynb, que es correcto, sino
# en su fuente de geometria:
#
#     data/raw/ecuador_provincias_poligonos.geojson
#
# cuyos poligonos tienen 5 vertices por provincia, es decir, la caja
# envolvente y no el contorno real. Este script reconstruye los mapas
# tomando la geometria de Natural Earth 1:10m, que si trae el contorno.
#
# QUE HACE
#   1. Rehace el dataset v6 (276 filas, 23 provincias, con elevacion).
#   2. Reproduce las predicciones fuera de pliegue con GroupKFold(5) y
#      los modelos RF v6 ya entrenados.
#   3. VALIDA el error por provincia contra las cifras publicadas en el
#      articulo. Si no coinciden, aborta: prefiere no dibujar nada a
#      dibujar algo que no corresponde al manuscrito.
#   4. Dibuja las tres coropletas con poligonos reales.
#
# DEPENDENCIAS:  pandas numpy scikit-learn matplotlib pyshp
# USO:           python3 regenerar_mapas.py [ruta/a/Tesis] [salida]
# =====================================================================
import sys
import os
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile                       # pyshp

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
import joblib

RAIZ = sys.argv[1] if len(sys.argv) > 1 else "../Tesis"
DEST = sys.argv[2] if len(sys.argv) > 2 else "figuras/mapas"
SHP = os.path.join(RAIZ, "data/external/ne_10m_admin_1_states_provinces")
os.makedirs(DEST, exist_ok=True)

# --- Elevacion de la capital provincial (msnm), identica al notebook ---
ELEVACION = {
    "Azuay": 2560, "Bolívar": 2668, "Cañar": 2518, "Carchi": 2957,
    "Chimborazo": 2754, "Cotopaxi": 2750, "El Oro": 4, "Esmeraldas": 5,
    "Guayas": 4, "Imbabura": 2225, "Loja": 2060, "Los Ríos": 7,
    "Manabí": 43, "Morona Santiago": 1070, "Napo": 511, "Orellana": 254,
    "Pastaza": 950, "Pichincha": 2850, "Santa Elena": 30,
    "Santo Domingo de los Tsáchilas": 550, "Sucumbíos": 295,
    "Tungurahua": 2577, "Zamora Chinchipe": 970, "Galápagos": 6,
}

# --- Cifras publicadas, para validar la reproduccion (en %) ------------
ESPERADO = {
    "solar": {"Santa Elena": 26.1, "Pichincha": 22.0, "Manabí": 18.8,
              "El Oro": 16.1, "Los Ríos": 14.6,
              "Cotopaxi": 4.0, "Bolívar": 4.7, "Pastaza": 5.3},
    "eolico": {"Santa Elena": 651, "Esmeraldas": 590, "Azuay": 519,
               "Napo": 3.0, "Sucumbíos": 4.6},
    "hidrico": {"Santa Elena": 125, "El Oro": 116, "Carchi": 103,
                "Cotopaxi": 15.3, "Zamora Chinchipe": 15.9,
                "Los Ríos": 15.8},
}


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


# =====================================================================
# 1. Dataset v6
# =====================================================================
df = pd.read_csv(os.path.join(RAIZ, "data/processed/final_dataset_ecuador.csv"))
assert len(df) == 288, f"se esperaban 288 filas, hay {len(df)}"
df["elevation"] = df["provincia"].map(ELEVACION)
assert df["elevation"].notna().all(), "falta elevacion en alguna provincia"

df = df[df["provincia"] != "Galápagos"].reset_index(drop=True)
assert len(df) == 276 and df["provincia"].nunique() == 23

F_SW = ["latitude", "longitude", "month", "temp_avg", "wind_speed_avg",
        "humidity_avg", "elevation"]
F_H = ["latitude", "longitude", "month", "temp_avg", "wind_speed_avg",
       "elevation"]

X_sw = SimpleImputer(strategy="median").fit_transform(df[F_SW].values)
X_h = SimpleImputer(strategy="median").fit_transform(df[F_H].values)
grupos = df["provincia"].values

CASOS = [
    ("solar",   "Solar",   X_sw, df["target_solar_kwh_per_m2"].values),
    ("eolico",  "Eólico",  X_sw, df["target_wind_mwh"].values),
    ("hidrico", "Hídrico", X_h,  df["target_hydro_mwh"].values),
]
MODELO = {"solar": "model_solar_v6.joblib",
          "eolico": "model_wind_v6.joblib",
          "hidrico": "model_hydro_v6.joblib"}

# =====================================================================
# 2. Predicciones fuera de pliegue + 3. validacion
# =====================================================================
mediana_eolico = float(np.median(df["target_wind_mwh"].values))
print(f"Mediana nacional eólica: {mediana_eolico:.4f} MWh/mes\n")

errores, etiquetas = {}, {}
for slug, nombre, X, y in CASOS:
    rf = joblib.load(os.path.join(RAIZ, "models", MODELO[slug]))
    oof = cross_val_predict(clone(rf), X, y,
                            cv=GroupKFold(n_splits=5), groups=grupos)

    t = pd.DataFrame({"provincia": grupos,
                      "residuo": np.abs(y - oof),
                      "target": y})
    agg = t.groupby("provincia").agg(mae=("residuo", "mean"),
                                     media=("target", "mean")).reset_index()

    if slug == "eolico":
        agg["valor"] = agg["mae"] / mediana_eolico * 100
        etiquetas[slug] = (f"Error (% mediana nacional = "
                           f"{mediana_eolico:.2f} MWh) [{nombre}]")
    else:
        agg["valor"] = agg["mae"] / agg["media"] * 100
        etiquetas[slug] = f"Error relativo MAE/media [{nombre}] (%)"

    errores[slug] = agg

    # --- validar contra el articulo ---------------------------------
    got = dict(zip(agg["provincia"], agg["valor"]))
    fallos = []
    for prov, esperado in ESPERADO[slug].items():
        real = got.get(prov)
        if real is None:
            fallos.append(f"{prov}: ausente")
        elif abs(real - esperado) > max(0.15, abs(esperado) * 0.012):
            fallos.append(f"{prov}: articulo {esperado} vs reproducido "
                          f"{real:.1f}")
    marca = "OK" if not fallos else "DISCREPA"
    print(f"[{marca}] {nombre}: {len(ESPERADO[slug])} provincias "
          f"contrastadas contra el articulo")
    for f in fallos:
        print("        ", f)
    if fallos:
        sys.exit("\nABORTADO: la reproduccion no coincide con el "
                 "manuscrito. No se generan mapas.")

# =====================================================================
# 4. Geometria real desde Natural Earth
# =====================================================================
sf = shapefile.Reader(SHP)
campos = [f[0] for f in sf.fields[1:]]
i_admin, i_name = campos.index("admin"), campos.index("name")

provincias = []
for sr in sf.shapeRecords():
    if sr.record[i_admin] != "Ecuador":
        continue
    nom = sr.record[i_name]
    if norm(nom).startswith("galapagos"):
        continue                      # fuera del entrenamiento, Tabla 4
    partes, pts = [], sr.shape.points
    idx = list(sr.shape.parts) + [len(pts)]
    for a, b in zip(idx[:-1], idx[1:]):
        partes.append(pts[a:b])
    provincias.append({"nombre": nom, "key": norm(nom), "partes": partes})

print(f"\nGeometría Natural Earth: {len(provincias)} provincias "
      f"continentales, {sum(len(p['partes']) for p in provincias)} anillos")
vert = sum(len(a) for p in provincias for a in p["partes"])
print(f"Vértices totales: {vert}  (el GeoJSON anterior tenía ~125)")


def dibujar(slug, titulo_barra, out):
    agg = errores[slug]
    val = {norm(r.provincia): r.valor for r in agg.itertuples()}
    vmax = float(np.percentile(list(val.values()), 90))   # recorte p90
    cmap = plt.get_cmap("RdYlGn_r")

    fig, ax = plt.subplots(figsize=(6.4, 7.6), dpi=300)
    for p in provincias:
        v = val.get(p["key"])
        color = "0.85" if v is None else cmap(min(v / vmax, 1.0))
        for anillo in p["partes"]:
            ax.add_patch(PathPatch(Path(anillo), facecolor=color,
                                   edgecolor="0.3", linewidth=0.45))
        if v is not None:
            mayor = max(p["partes"], key=len)
            xs = [q[0] for q in mayor]
            ys = [q[1] for q in mayor]
            ax.annotate(p["nombre"],
                        ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
                        ha="center", va="center", fontsize=4.6,
                        fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.3,
                                                    foreground="white")])

    todos = [q for p in provincias for a in p["partes"] for q in a]
    xs = [q[0] for q in todos]
    ys = [q[1] for q in todos]
    m = 0.25
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.045, pad=0.02, shrink=0.9, extend="max")
    cb.set_label(titulo_barra, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  escrito: {out}   (escala 0–{vmax:.1f}, recorte p90)")


print()
for slug, _, _, _ in CASOS:
    dibujar(slug, etiquetas[slug], os.path.join(DEST, f"residuos_{slug}.png"))

print("\nListo. Ahora: python3 build_figuras.py  para recomponer la Figura 1.")

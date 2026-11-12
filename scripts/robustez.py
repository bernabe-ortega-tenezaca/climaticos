#!/usr/bin/env python3
# =====================================================================
# robustez.py — análisis de robustez sobre el pipeline v6
#
# No reentrena ni modifica los modelos publicados: reutiliza el dataset
# final, los RF v6 guardados y los hiperparámetros de metadata_v6.json.
#
#   0. Reproduce la Tabla 3 (control: debe coincidir con el artículo).
#   1. Validación espacial ANIDADA: el ajuste de hiperparámetros se repite
#      dentro de cada pliegue externo, sin ver las provincias retenidas.
#   2. Leave-one-province-out (23 pliegues): R² agregado, bootstrap por
#      provincias y prueba de Wilcoxon pareada contra la línea base mensual.
#   3. Sensibilidad a la partición: 200 asignaciones aleatorias de
#      provincias a 5 pliegues.
#   4. Mecanismo: modelo que solo ve identificadores geográficos y mes.
#
# Uso (desde la raíz del repositorio): python3 scripts/robustez.py .
# =====================================================================
import sys, os, json, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (GroupKFold, GridSearchCV, LeaveOneGroupOut,
                                     cross_val_score, cross_val_predict,
                                     train_test_split)
import joblib
warnings.filterwarnings("ignore")

RAIZ = sys.argv[1] if len(sys.argv) > 1 else "."
RS = 42
ELEV = {"Azuay":2560,"Bolívar":2668,"Cañar":2518,"Carchi":2957,"Chimborazo":2754,
 "Cotopaxi":2750,"El Oro":4,"Esmeraldas":5,"Guayas":4,"Imbabura":2225,"Loja":2060,
 "Los Ríos":7,"Manabí":43,"Morona Santiago":1070,"Napo":511,"Orellana":254,
 "Pastaza":950,"Pichincha":2850,"Santa Elena":30,
 "Santo Domingo de los Tsáchilas":550,"Sucumbíos":295,"Tungurahua":2577,
 "Zamora Chinchipe":970,"Galápagos":6}

class MonthMean(BaseEstimator, RegressorMixin):
    def __init__(self, col=2): self.col = col
    def fit(self, X, y):
        m = X[:, self.col].astype(int); self.g_ = float(y.mean())
        self.mm_ = {int(k): float(y[m == k].mean()) for k in np.unique(m)}
        return self
    def predict(self, X):
        return np.array([self.mm_.get(int(k), self.g_) for k in X[:, self.col].astype(int)])

df = pd.read_csv(os.path.join(RAIZ, "data/processed/final_dataset_ecuador.csv"))
df["elevation"] = df["provincia"].map(ELEV)
df = df[df["provincia"] != "Galápagos"].reset_index(drop=True)
F_SW = ["latitude","longitude","month","temp_avg","wind_speed_avg","humidity_avg","elevation"]
F_H  = ["latitude","longitude","month","temp_avg","wind_speed_avg","elevation"]
g = df["provincia"].values
meta = json.load(open(os.path.join(RAIZ, "models/metadata_v6.json")))
HP = meta["hiperparametros"]
CASOS = {"solar": (df[F_SW].values, df["target_solar_kwh_per_m2"].values, F_SW),
         "wind":  (df[F_SW].values, df["target_wind_mwh"].values, F_SW),
         "hydro": (df[F_H].values,  df["target_hydro_mwh"].values, F_H)}
NOMBRE = {"solar": "Solar", "wind": "Eólico", "hydro": "Hídrico"}

def modelos(mk):
    return {"LR": LinearRegression(),
            "RF": RandomForestRegressor(random_state=RS, n_jobs=-1, **HP[f"rf_{mk}"]),
            "GB": GradientBoostingRegressor(random_state=RS, **HP[f"gb_{mk}"])}

res = {}
# ---------------------------------------------------------------- 0
print("=" * 72, "\n0. CONTROL: reproducción de la Tabla 3 (GroupKFold 5)\n" + "=" * 72)
gkf5 = GroupKFold(n_splits=5)
for mk, (X, y, _) in CASOS.items():
    for a, m in {"Base_media": DummyRegressor(), "Base_mensual": MonthMean(), **modelos(mk)}.items():
        sc = cross_val_score(m, X, y, cv=gkf5, groups=g, scoring="r2")
        res[(mk, a, "gkf5")] = (sc.mean(), sc.std())
        print(f"  {NOMBRE[mk]:8} {a:13} R²={sc.mean():+.4f} ± {sc.std():.4f}")

# ---------------------------------------------------------------- 1
print("\n" + "=" * 72, "\n1. VALIDACIÓN ESPACIAL ANIDADA (externo GKF5, interno GKF3)\n" + "=" * 72)
P_RF = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
P_GB = {"n_estimators": [100, 200], "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2], "min_samples_split": [2, 5]}
for mk, (X, y, _) in CASOS.items():
    for a, est, grid in [("RF", RandomForestRegressor(random_state=RS, n_jobs=-1), P_RF),
                         ("GB", GradientBoostingRegressor(random_state=RS), P_GB)]:
        sc = []
        for tr, te in gkf5.split(X, y, g):
            gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=3), scoring="r2", n_jobs=-1)
            gs.fit(X[tr], y[tr], groups=g[tr])
            sc.append(r2_score(y[te], gs.predict(X[te])))
        sc = np.array(sc); res[(mk, a, "nested")] = (sc.mean(), sc.std())
        b = res[(mk, a, "gkf5")]
        print(f"  {NOMBRE[mk]:8} {a}  anidada R²={sc.mean():+.4f} ± {sc.std():.4f}   "
              f"(publicada {b[0]:+.4f};  Δ={sc.mean()-b[0]:+.4f})")

# ---------------------------------------------------------------- 2
print("\n" + "=" * 72, "\n2. LEAVE-ONE-PROVINCE-OUT (23 pliegues) + bootstrap por provincias\n" + "=" * 72)
logo = LeaveOneGroupOut(); provs = np.unique(g); rng = np.random.default_rng(RS)
for mk, (X, y, _) in CASOS.items():
    oof = {a: cross_val_predict(m, X, y, cv=logo, groups=g)
           for a, m in {"Base_mensual": MonthMean(), **modelos(mk)}.items()}
    idx = {p: np.where(g == p)[0] for p in provs}
    mae_p = {a: np.array([mean_absolute_error(y[idx[p]], o[idx[p]]) for p in provs]) for a, o in oof.items()}
    print(f"\n  --- {NOMBRE[mk]}")
    for a, o in oof.items():
        boot = []
        for _ in range(2000):
            s = np.concatenate([idx[p] for p in rng.choice(provs, len(provs))])
            boot.append(r2_score(y[s], o[s]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        linea = f"  {a:13} R² agregado={r2_score(y, o):+.3f}  IC95 [{lo:+.3f}; {hi:+.3f}]  MAE={mean_absolute_error(y, o):.3f}"
        res[(mk, a, "lopo")] = (r2_score(y, o), lo, hi, mean_absolute_error(y, o))
        if a != "Base_mensual":
            d = mae_p["Base_mensual"] - mae_p[a]
            w = wilcoxon(mae_p[a], mae_p["Base_mensual"])
            res[(mk, a, "wilcoxon")] = (int((d > 0).sum()), float(w.pvalue))
            linea += f"   mejora a la base mensual en {(d>0).sum()}/23 provincias, Wilcoxon p={w.pvalue:.4f}"
        print(linea)

# ---------------------------------------------------------------- 3
print("\n" + "=" * 72, "\n3. SENSIBILIDAD A LA PARTICIÓN: 200 asignaciones aleatorias a 5 pliegues\n" + "=" * 72)
rng = np.random.default_rng(RS)
for mk, (X, y, _) in CASOS.items():
    for a in ("RF", "GB"):
        m = modelos(mk)[a]; medias = []
        for _ in range(200):
            perm = rng.permutation(provs); fold = {p: i % 5 for i, p in enumerate(perm)}
            f = np.array([fold[p] for p in g]); sc = []
            for k in range(5):
                mm = clone(m).fit(X[f != k], y[f != k]); sc.append(r2_score(y[f == k], mm.predict(X[f == k])))
            medias.append(np.mean(sc))
        q = np.percentile(medias, [5, 25, 50, 75, 95]); res[(mk, a, "rep")] = tuple(q)
        print(f"  {NOMBRE[mk]:8} {a}  mediana={q[2]:+.3f}  RIC [{q[1]:+.3f}; {q[3]:+.3f}]  p5–p95 [{q[0]:+.3f}; {q[4]:+.3f}]"
              f"   (partición publicada: {res[(mk,a,'gkf5')][0]:+.3f})")

# ---------------------------------------------------------------- 4
print("\n" + "=" * 72, "\n4. MECANISMO: RF que solo ve latitud, longitud, elevación y mes\n" + "=" * 72)
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS)
ID = ["latitude", "longitude", "elevation", "month"]; Xi = df[ID].values
for mk, (X, y, _) in CASOS.items():
    rf = RandomForestRegressor(random_state=RS, n_jobs=-1, **HP[f"rf_{mk}"])
    ho = r2_score(y[te], clone(rf).fit(Xi[tr], y[tr]).predict(Xi[te]))
    sp = cross_val_score(rf, Xi, y, cv=gkf5, groups=g, scoring="r2")
    full = r2_score(y[te], clone(rf).fit(X[tr], y[tr]).predict(X[te]))
    res[(mk, "RF_id", "mec")] = (ho, sp.mean(), full)
    print(f"  {NOMBRE[mk]:8} solo identificadores: aleatoria R²={ho:+.3f} | espacial R²={sp.mean():+.3f}"
          f"   (RF completo, aleatoria: {full:+.3f})")

json.dump({"|".join(k): [float(x) for x in v] for k, v in res.items()},
          open(os.path.join(RAIZ, "data/processed/robustez_resultados.json"), "w"), indent=1, ensure_ascii=False)
print("\n[OK] data/processed/robustez_resultados.json")

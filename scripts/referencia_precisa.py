#!/usr/bin/env python3
"""Recalcula con precisión completa las cifras de las Tablas 2, 3 y 5.
Los CSV del pipeline guardan cuatro decimales; redondear de nuevo a tres
puede desplazar el último dígito. Este script reproduce los modelos v6
(mismos hiperparámetros y semilla) y guarda los valores sin redondear."""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate
warnings.filterwarnings("ignore")
T = sys.argv[1] if len(sys.argv) > 1 else "."; RS = 42
ELEV = {"Azuay":2560,"Bolívar":2668,"Cañar":2518,"Carchi":2957,"Chimborazo":2754,"Cotopaxi":2750,
 "El Oro":4,"Esmeraldas":5,"Guayas":4,"Imbabura":2225,"Loja":2060,"Los Ríos":7,"Manabí":43,
 "Morona Santiago":1070,"Napo":511,"Orellana":254,"Pastaza":950,"Pichincha":2850,"Santa Elena":30,
 "Santo Domingo de los Tsáchilas":550,"Sucumbíos":295,"Tungurahua":2577,"Zamora Chinchipe":970,"Galápagos":6}

class MonthMean(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        m = X[:, 2].astype(int); self.g_ = float(y.mean())
        self.mm_ = {int(k): float(y[m == k].mean()) for k in np.unique(m)}; return self
    def predict(self, X):
        return np.array([self.mm_.get(int(k), self.g_) for k in X[:, 2].astype(int)])

full = pd.read_csv(f"{T}/data/processed/final_dataset_ecuador.csv"); full["elevation"] = full.provincia.map(ELEV)
gal = full[full.provincia == "Galápagos"]; df = full[full.provincia != "Galápagos"].reset_index(drop=True)
F_SW = ["latitude","longitude","month","temp_avg","wind_speed_avg","humidity_avg","elevation"]
F_H = [f for f in F_SW if f != "humidity_avg"]
HP = json.load(open(f"{T}/models/metadata_v6.json"))["hiperparametros"]
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS); g = df.provincia.values
C = {"solar": (F_SW, "target_solar_kwh_per_m2"), "wind": (F_SW, "target_wind_mwh"), "hydro": (F_H, "target_hydro_mwh")}
out, M = {}, {}
for mk, (F, t) in C.items():
    X, y = df[F].values, df[t].values
    M[mk] = {"LR": LinearRegression().fit(X[tr], y[tr]),
             "RF": RandomForestRegressor(random_state=RS, n_jobs=-1, **HP[f"rf_{mk}"]).fit(X[tr], y[tr]),
             "GB": GradientBoostingRegressor(random_state=RS, **HP[f"gb_{mk}"]).fit(X[tr], y[tr])}
    for a, m in M[mk].items():
        p = m.predict(X[te])
        out[f"holdout|{mk}|{a}"] = [mean_absolute_error(y[te], p), float(np.sqrt(mean_squared_error(y[te], p))), r2_score(y[te], p)]
        if a != "LR":
            pg = m.predict(gal[F].values); yg = gal[t].values
            out[f"ood|{mk}|{a}"] = [mean_absolute_error(yg, pg), float(np.sqrt(mean_squared_error(yg, pg))), r2_score(yg, pg)]
    for a, m in {"Base_media": DummyRegressor(), "Base_mensual": MonthMean(), **M[mk]}.items():
        cv = cross_validate(m, X, y, cv=GroupKFold(5), groups=g, scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})
        out[f"gkf|{mk}|{a}"] = [cv["test_r2"].mean(), cv["test_r2"].std(), -cv["test_mae"].mean()]
rng = np.random.default_rng(42); n = len(te)          # misma secuencia que el notebook
for mk, (F, t) in C.items():
    X, y = df[F].values, df[t].values; yt = y[te]
    for a in ("LR", "RF", "GB"):
        yp = M[mk][a].predict(X[te]); r = []
        for _ in range(1000):
            i = rng.integers(0, n, size=n); r.append(r2_score(yt[i], yp[i]))
            mean_absolute_error(yt[i], yp[i]); np.sqrt(mean_squared_error(yt[i], yp[i]))
        out[f"boot|{mk}|{a}"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
json.dump({k: [float(x) for x in v] for k, v in out.items()}, open(T + "/data/processed/referencia_precisa.json", "w"), indent=1)
print("[OK] referencia_precisa.json:", len(out), "entradas")

from __future__ import annotations

import pandas as pd
import numpy as np
import pvlib
import pytz


def calculate_solar_potential(monthly_pvgis_data: pd.DataFrame) -> float:
    if monthly_pvgis_data.empty:
        return 0.0

    latitude = monthly_pvgis_data['latitude'].iloc[0]
    longitude = monthly_pvgis_data['longitude'].iloc[0]

    # Tilt mínimo de 5° para evitar ángulos negativos en el ecuador
    surface_tilt = max(abs(latitude) * 0.9, 5.0)
    # Azimut óptimo según hemisferio: 0° (norte) para hemisferio sur, 180° (sur) para norte
    surface_azimuth = 0 if latitude < 0 else 180

    tz = pytz.timezone('America/Guayaquil')
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=tz
    )

    idx_local = monthly_pvgis_data.index.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    idx_local = idx_local.dropna()

    if idx_local.empty:
        return 0.0

    monthly_local = monthly_pvgis_data.loc[idx_local]

    solpos = location.get_solarposition(monthly_local.index)

    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=monthly_local['dni'],
        ghi=monthly_local['ghi'],
        dhi=monthly_local['dhi']
    )

    # poa_global en W/m²; suma → Wh/m²; /1000 → kWh/m²; *0.15 → eficiencia panel 15%
    energy_kwh = (poa_irradiance['poa_global'].sum() / 1000) * 0.15
    return energy_kwh


def power_curve(wind_speed: float) -> float:
    if wind_speed < 3 or wind_speed > 25:
        return 0.0
    if wind_speed <= 12:
        return (wind_speed / 12) ** 3 * 2.0
    return 2.0


def get_zona(provincia: str) -> str:
    """
    Clasifica una provincia ecuatoriana en su zona climática.
    Útil para imputación regionalizada y coeficientes Hellman.
    """
    zonas_climaticas = {
        'Costa': ['Esmeraldas', 'Manabí', 'Santa Elena', 'Guayas',
                  'El Oro', 'Los Ríos', 'Santo Domingo de los Tsáchilas'],
        'Sierra': ['Carchi', 'Imbabura', 'Pichincha', 'Cotopaxi',
                   'Tungurahua', 'Bolívar', 'Chimborazo', 'Cañar',
                   'Azuay', 'Loja'],
        'Amazonía': ['Sucumbíos', 'Napo', 'Orellana', 'Pastaza',
                     'Morona Santiago', 'Zamora Chinchipe'],
        'Insular': ['Galápagos']
    }
    for zona, provincias in zonas_climaticas.items():
        if provincia in provincias:
            return zona
    return 'Desconocido'


def get_hellman_alpha(provincia: str = '') -> float:
    """
    Coeficiente de Hellman regionalizado por zona geográfica de Ecuador.
    Fuente: Manwell, J.F. et al. (2009). Wind Energy Explained. Wiley.
    """
    zonas_alpha = {
        'Costa': 0.14,       # Terreno abierto/plano
        'Sierra': 0.25,      # Terreno montañoso
        'Amazonía': 0.20,    # Bosque denso
        'Insular': 0.10,     # Sobre agua/islas
    }
    zona = get_zona(provincia)
    return zonas_alpha.get(zona, 0.14)


def calculate_wind_potential(
    monthly_weather_data: pd.DataFrame,
    provincia: str = ''
) -> float:
    if monthly_weather_data.empty:
        return 0.0

    wind_speed_2m = monthly_weather_data['wspd']
    height_ref, height_target = 2, 100
    alpha = get_hellman_alpha(provincia)
    wind_speed_100m = wind_speed_2m * (height_target / height_ref) ** alpha

    power_mw = wind_speed_100m.apply(power_curve)
    energy_mwh = power_mw.sum()
    return energy_mwh


def calculate_hydro_potential(
    monthly_weather_data: pd.DataFrame,
    area_km2: float = 500,
    height_m: float = 200
) -> float:
    if monthly_weather_data.empty:
        return 0.0

    precip_mm = monthly_weather_data['prcp'].sum()
    humidity = monthly_weather_data['relative_humidity'].mean() / 100
    temp = monthly_weather_data['tavg'].mean()

    # Coeficiente de escorrentía empírico ad-hoc (limitación conocida:
    # no validado con datos hidrológicos de cuencas ecuatorianas).
    # Valores de referencia: 0.3 (base), 0.4×hum (aporte por humedad),
    # -0.002×T (penalización por evaporación). Clip en [0.1, 0.85].
    runoff_coeff = 0.3 + (0.4 * humidity) - (0.002 * max(temp, 0))
    runoff_coeff = np.clip(runoff_coeff, 0.1, 0.85)

    precip_m = precip_mm / 1000
    area_m2 = area_km2 * 1e6
    dias_mes = monthly_weather_data.shape[0]
    segundos_mes = dias_mes * 24 * 3600

    volumen_m3 = precip_m * area_m2 * runoff_coeff
    caudal_m3s = volumen_m3 / segundos_mes if segundos_mes > 0 else 0

    rho = 1000
    g = 9.81
    eta = 0.85

    potencia_mw = (rho * g * caudal_m3s * height_m * eta) / 1e6
    energia_mwh = potencia_mw * (dias_mes * 24)

    return round(energia_mwh, 4)

"""Tests unitarios para funciones de cálculo de potencial energético."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.preprocessing.targets import (
    power_curve,
    calculate_wind_potential,
    calculate_hydro_potential,
    calculate_solar_potential,
)


def test_power_curve_below_cutin() -> None:
    assert power_curve(2.0) == 0.0


def test_power_curve_above_cutout() -> None:
    assert power_curve(26.0) == 0.0


def test_power_curve_rated() -> None:
    assert power_curve(12.0) == 2.0


def test_power_curve_partial() -> None:
    p = power_curve(6.0)
    assert 0.2 < p < 0.4


def test_calculate_wind_potential_empty() -> None:
    df = pd.DataFrame()
    assert calculate_wind_potential(df) == 0.0


def test_calculate_wind_potential_no_wind() -> None:
    df = _make_weather_df(wspd=0.0, prcp=0.0, hum=50.0, temp=20.0, days=30)
    assert calculate_wind_potential(df) == 0.0


def test_calculate_wind_potential_fair_wind() -> None:
    df = _make_weather_df(wspd=6.0, prcp=0.0, hum=50.0, temp=20.0, days=30)
    result = calculate_wind_potential(df)
    assert result > 0.0


def test_calculate_hydro_potential_empty() -> None:
    df = pd.DataFrame()
    assert calculate_hydro_potential(df) == 0.0


def test_calculate_hydro_potential_dry() -> None:
    df = _make_weather_df(wspd=0.0, prcp=0.0, hum=50.0, temp=25.0, days=30)
    result = calculate_hydro_potential(df, area_km2=500, height_m=200)
    assert result == 0.0


def test_calculate_hydro_potential_rainy() -> None:
    df = _make_weather_df(wspd=0.0, prcp=5.0, hum=80.0, temp=25.0, days=30)
    result = calculate_hydro_potential(df, area_km2=500, height_m=200)
    assert result > 0.0


def test_calculate_solar_potential_empty() -> None:
    df = pd.DataFrame()
    assert calculate_solar_potential(df) == 0.0


def _make_weather_df(
    wspd: float,
    prcp: float,
    hum: float,
    temp: float,
    days: int,
) -> pd.DataFrame:
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    return pd.DataFrame({
        'wspd': [wspd] * days,
        'prcp': [prcp] * days,
        'relative_humidity': [hum] * days,
        'tavg': [temp] * days,
    }, index=dates)

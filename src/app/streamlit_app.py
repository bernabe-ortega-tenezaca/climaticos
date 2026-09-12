from pathlib import Path
import logging
import os

import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

def create_pdf_report(provincia: str, mes: int, prediction: dict) -> io.BytesIO:
    etiquetas = {
        "potencial_solar_kwh_por_m2": "Potencial Solar (kWh/m²)",
        "potencial_eolico_mwh": "Potencial Eolico (MWh)",
        "potencial_hidrico_mwh": "Potencial Hidrico (MWh)",
    }
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    p.setFont("Helvetica-Bold", 16)
    p.drawString(inch, height - inch, "Reporte de Potencial Energetico")
    p.setFont("Helvetica", 12)
    p.drawString(inch, height - 1.5 * inch, f"Provincia: {provincia}")
    p.drawString(inch, height - 1.75 * inch, f"Mes: {mes}")
    y_position = height - 2.5 * inch
    p.drawString(inch, y_position, "Predicciones:")
    y_position -= 0.3 * inch
    for key, value in prediction.items():
        label = etiquetas.get(key, key)
        p.drawString(1.2 * inch, y_position, f"{label}: {value}")
        y_position -= 0.25 * inch
    p.save()
    buffer.seek(0)
    return buffer

st.set_page_config(
    page_title="Atlas Energetico de Ecuador",
    page_icon=":zap:",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_data() -> tuple:
    try:
        shapefile_path = DATA_DIR / 'external' / 'ne_10m_admin_1_states_provinces.shp'
        gdf_poligonos = gpd.read_file(shapefile_path)
        gdf_ecuador = gdf_poligonos[gdf_poligonos['admin'] == 'Ecuador'].copy()
        gdf_ecuador.rename(columns={'name': 'provincia'}, inplace=True)
        dataset_path = DATA_DIR / 'processed' / 'final_dataset_ecuador.csv'
        df_final = pd.read_csv(dataset_path)
        logger.info("Datos cargados correctamente")
        return gdf_ecuador, df_final
    except Exception as e:
        logger.error("Error al cargar datos: %s", e)
        st.error(f"Error al cargar los datos: {e}")
        return None, None

gdf_ecuador, df_final = load_data()

if gdf_ecuador is None or df_final is None:
    st.stop()

provincias_disponibles = sorted(gdf_ecuador['provincia'].unique())
meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
periodos = [f"{m}-{a}" for a in range(2021, 2025) for m in meses[:12]]

st.title("Atlas Energetico de Ecuador")
st.markdown("Visualizacion geoespacial del potencial de energias renovables.")

tab_mapa, tab_datos = st.tabs(["Mapa Interactivo", "Exploracion de Datos"])

with tab_mapa:
    col_mapa, col_resultados = st.columns([3, 1])
    with col_mapa:
        m = folium.Map(location=[-1.5, -78.5], zoom_start=6, tiles="CartoDB positron")
        for _, row in gdf_ecuador.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 1, 'fillOpacity': 0.3},
                highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.6},
                tooltip=folium.Tooltip(row['provincia']),
            ).add_to(m)
        map_data = st_folium(m, height=500, width=None)
        if map_data and map_data.get("last_object_clicked"):
            clicked_point = map_data["last_object_clicked"]
            lat_click, lon_click = clicked_point["lat"], clicked_point["lng"]
            distances = gdf_ecuador.distance(gpd.points_from_xy([lon_click], [lat_click]).iloc[0])
            nearest_idx = distances.idxmin()
            st.session_state.selected_province = gdf_ecuador.loc[nearest_idx, 'provincia']

    with col_resultados:
        st.subheader("Provincia Seleccionada")
        if 'selected_province' in st.session_state:
            provincia = st.session_state.selected_province
            st.success(f"**{provincia}**")
            datos_prov = df_final[df_final['provincia'] == provincia]
            if not datos_prov.empty:
                promedios = datos_prov[['target_solar_kwh_per_m2', 'target_wind_mwh', 'target_hydro_mwh']].mean()
                st.metric("Potencial Solar Promedio", f"{promedios['target_solar_kwh_per_m2']:.2f} kWh/m²")
                st.metric("Potencial Eolico Promedio", f"{promedios['target_wind_mwh']:.2f} MWh")
                st.metric("Potencial Hidrico Promedio", f"{promedios['target_hydro_mwh']:.2f} MWh")
        else:
            st.info("Haz clic en el mapa para seleccionar una provincia.")

st.sidebar.subheader("Parametros Climaticos y Temporales")
selected_month = st.sidebar.slider("Mes del Ano", 1, 12, value=6)
temp_avg = st.sidebar.slider("Temperatura Promedio (°C)", -10.0, 40.0, value=15.0)
wind_speed_avg = st.sidebar.slider("Velocidad del Viento Promedio (m/s)", 0.0, 30.0, value=5.0)
humidity_avg = st.sidebar.slider("Humedad Relativa Promedio (%)", 0.0, 100.0, value=70.0)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("Predecir Potencial", type="primary", use_container_width=True)
generate_pdf_button = st.sidebar.button("Generar Reporte PDF", type="secondary", use_container_width=True)

if predict_button:
    if 'selected_province' not in st.session_state:
        st.sidebar.warning("Por favor, selecciona una provincia en el mapa primero.")
    else:
        with st.spinner("Realizando prediccion..."):
            province_centroid = gdf_ecuador[gdf_ecuador['provincia'] == st.session_state.selected_province].geometry.centroid.iloc[0]
            input_data = {
                "latitude": province_centroid.y,
                "longitude": province_centroid.x,
                "month": int(selected_month),
                "temp_avg": float(temp_avg),
                "wind_speed_avg": float(wind_speed_avg),
                "humidity_avg": float(humidity_avg)
            }
            try:
                response = requests.post(API_URL, json=input_data)
                if "error" in response.text:
                    st.error(f"Error desde la API: {response.json().get('detail', 'Error desconocido')}")
                else:
                    response.raise_for_status()
                    prediction_result = response.json()
                    st.session_state.last_prediction = prediction_result
                    st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar a la API.")
            except Exception as e:
                st.error(f"Ocurrio un error: {e}")

if generate_pdf_button:
    if 'selected_province' not in st.session_state:
        st.sidebar.warning("Por favor, selecciona una provincia primero.")
    elif 'last_prediction' not in st.session_state:
        st.sidebar.warning("Por favor, realiza una prediccion antes de generar el reporte.")
    else:
        with st.spinner("Generando reporte PDF..."):
            pdf_buffer = create_pdf_report(
                st.session_state.selected_province,
                selected_month,
                st.session_state.last_prediction
            )
            st.sidebar.download_button(
                label="Descargar Reporte PDF",
                data=pdf_buffer,
                file_name=f"reporte_energetico_{st.session_state.selected_province}.pdf",
                mime="application/pdf"
            )

st.sidebar.markdown("---")
st.sidebar.markdown("### Informacion")
st.sidebar.info("Esta aplicacion utiliza modelos de Machine Learning entrenados con datos historicos de Ecuador.")

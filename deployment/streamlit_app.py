# deployment/streamlit_app.py (Versión Final con Diseño tipo Bootstrap)

import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import unicodedata
import plotly.express as px
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- CONFIGURACIÓN DE PÁGINA Y CSS ---
st.set_page_config(
    page_title="Potencial Energético - Ecuador",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el CSS personalizado
def local_css(file_name):
    # Construye la ruta al archivo de forma robusta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/style.css")

# --- FUNCIONES AUXILIARES ---
def normalizar_nombre(nombre):
    return unicodedata.normalize('NFKD', nombre).encode('ascii', errors='ignore').decode('utf-8')

def llamar_api(provincia: str, energia: str):
    api_url = "http://127.0.0.1:8000/predecir_por_provincia"
    params = {"provincia": provincia, "energia": energia}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API: {e}")
        st.warning("Asegúrate de que el servidor de FastAPI está corriendo en http://127.0.0.1:8000")
        return None

@st.cache_resource
def cargar_datos_visualizacion():
    recursos = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'dataset_para_app_completo.csv')
    
    recursos['df'] = pd.read_csv(csv_path)
    recursos['df']['provincia_std'] = recursos['df']['provincia'].apply(normalizar_nombre)

    shapefile_path = os.path.join(current_dir, '..', 'data', 'geographic', 'ne_10m_admin_1_states_provinces.shp')
    gdf = gpd.read_file(shapefile_path)
    recursos['gdf'] = gdf[gdf['admin'] == 'Ecuador'].copy()
    recursos['gdf']['name_std'] = recursos['gdf']['name'].apply(normalizar_nombre)
    return recursos

recursos_vis = cargar_datos_visualizacion()

def plot_importancia(modelo_path, energia_tipo, color_palette):
    try:
        modelo = joblib.load(modelo_path)
        preprocessor = modelo.named_steps['preprocessor']
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['provincia'])
        numerical_features = ['altitud', 'temp_promedio_anual_C', 'viento_promedio_anual_ms', 'potencial_hidrico_proxy_mm']
        all_feature_names = numerical_features + list(ohe_feature_names)
        importances = pd.Series(modelo.named_steps['regressor'].feature_importances_, index=all_feature_names)
        importances = importances.sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=importances.values, y=importances.index, palette=color_palette, ax=ax)
        ax.set_title(f'Top 10 Variables Más Importantes para {energia_tipo}', fontsize=16)
        ax.set_xlabel('Importancia'); ax.set_ylabel('Variable')
        return fig
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo '{modelo_path}'. Error: {e}")
        return None

# --- LAYOUT DE LA APLICACIÓN (USANDO CARDS) ---

# --- CARD 1: TÍTULO PRINCIPAL ---
st.markdown("""
<div class="card">
    <h1>⚡ Potencial Energético Renovable de Ecuador</h1>
    <p style="text-align: center; font-size: 1.1rem; color: #6c757d;">
        Una plataforma interactiva que utiliza Inteligencia Artificial para analizar y predecir el potencial de generación de energía solar, eólica e hídrica en las provincias de Ecuador.
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR PARA CONTROLES ---
st.sidebar.header("Panel de Control")
tipo_energia_seleccionado = st.sidebar.selectbox("Tipo de Energía", ["Solar", "Eólica", "Hídrica"])
provincias_unicas = sorted(recursos_vis['df']['provincia'].unique())
provincia_seleccionada = st.sidebar.selectbox("Selecciona una Provincia", provincias_unicas)

# --- CARD 2: CENTRO DE PREDICCIONES ---
st.markdown(f'<div class="card"><h2 class="card-title">🎯 Centro de Predicciones</h2><p>Análisis del potencial <strong>{tipo_energia_seleccionado.lower()}</strong> para <strong>{provincia_seleccionada}</strong>.</p></div>', unsafe_allow_html=True)

resultado_api = llamar_api(provincia_seleccionada, tipo_energia_seleccionado)

if resultado_api:
    col1, col2, col3 = st.columns(3)
    prediccion_energia = resultado_api['energia_anual_predicha']['valor']
    unidad_energia = resultado_api['energia_anual_predicha']['unidad']
    prediccion_viabilidad = resultado_api['viabilidad_predicha']
    col1.metric(f"Energía Anual ({unidad_energia})", f"{prediccion_energia:,.2f}")
    if 'potencia_media_predicha' in resultado_api:
        potencia_kw = resultado_api['potencia_media_predicha']['valor']
        col2.metric("Potencia Media (kW)", f"{potencia_kw:,.2f}")
    col3.metric("Viabilidad", prediccion_viabilidad)

# --- CARD 3: MAPA NACIONAL ---
st.markdown(f'<div class="card"><h2 class="card-title">🗺️ Potencial a Nivel Nacional</h2><p>Mapa interactivo del potencial de energía <strong>{tipo_energia_seleccionado.lower()}</strong> en Ecuador.</p></div>', unsafe_allow_html=True)
def generar_mapa_app(tipo_energia, df, gdf):
    if tipo_energia == 'Solar': valor_col = 'energia_solar_anual_kwh'; color_scale = "YlOrRd"
    elif tipo_energia == 'Eólica': valor_col = 'energia_eolica_anual_kwh'; color_scale = "Viridis"
    else: valor_col = 'potencial_hidrico_proxy_mm'; color_scale = "Blues"
    energia_promedio = df.groupby('provincia_std')[valor_col].mean().reset_index()
    mapa_datos = gdf.merge(energia_promedio, left_on='name_std', right_on='provincia_std', how='left')
    fig_mapa = px.choropleth_mapbox(mapa_datos, geojson=mapa_datos.geometry, locations=mapa_datos.index, color=valor_col, hover_name='name', hover_data={valor_col: ':.2f'}, color_continuous_scale=color_scale, mapbox_style="carto-positron", zoom=5.5, center={"lat": -1.831, "lon": -78.183}, opacity=0.7)
    fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig_mapa
fig_mapa = generar_mapa_app(tipo_energia_seleccionado, recursos_vis['df'], recursos_vis['gdf'])
if fig_mapa: st.plotly_chart(fig_mapa, use_container_width=True)

# --- CARD 4: RANKING ---
st.markdown(f'<div class="card"><h2 class="card-title">📊 Ranking de Potencial</h2><p>Comparación del potencial de energía <strong>{tipo_energia_seleccionado.lower()}</strong> entre todas las provincias.</p></div>', unsafe_allow_html=True)
if tipo_energia_seleccionado == 'Solar': valor_col = 'energia_solar_anual_kwh'; unidad = "kWh"
elif tipo_energia_seleccionado == 'Eólica': valor_col = 'energia_eolica_anual_kwh'; unidad = "kWh"
else: valor_col = 'potencial_hidrico_proxy_mm'; unidad = "mm"
df_comparativo = recursos_vis['df'].groupby('provincia')[valor_col].mean().sort_values(ascending=False)
fig_bar = px.bar(df_comparativo, x=df_comparativo.values, y=df_comparativo.index, orientation='h', labels={'x': f'Potencial Anual Promedio ({unidad})', 'y': 'Provincia'}, title=f"Ranking de Potencial {tipo_energia_seleccionado}")
fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_bar, use_container_width=True)

# --- CARD 5: ANÁLISIS EXPLORATORIO ---
st.markdown('<div class="card"><h2 class="card-title">📈 Análisis Exploratorio y Resultados del Modelo</h2><p>Visualizaciones clave que fundamentan las predicciones del modelo y revelan patrones importantes.</p></div>', unsafe_allow_html=True)

# Sub-card para el Mapa de Calor
st.markdown('<h3 style="color: #495057;">Figura 1: Matriz de Correlación de Variables</h3>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 0.9em; color: #6c757d;">*Figura 1. Matriz de correlación de Pearson entre las variables climáticas, geográficas y de potencial energético. Fuente: Elaboración propia.*</p>', unsafe_allow_html=True)
df_numerico = recursos_vis['df'].select_dtypes(include=np.number)
corr_matrix = df_numerico.corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax)
ax.set_title('Matriz de Correlación', fontsize=16)
st.pyplot(fig); plt.close(fig)

# Sub-card para los Boxplots
st.markdown('<h3 style="color: #495057;">Figura 2: Distribución de Variables Clave por Viabilidad</h3>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 0.9em; color: #6c757d;">*Figura 2. Diagramas de caja que comparan la distribución de variables clave entre las categorías de viabilidad. Fuente: Elaboración propia.*</p>', unsafe_allow_html=True)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Relación entre Variables Clave y Viabilidad Energética', fontsize=20)
sns.boxplot(ax=axes[0], x=recursos_vis['df']['viabilidad_solar'], y=recursos_vis['df']['temp_promedio_anual_C'], palette='YlOrRd'); axes[0].set_title('Solar: Temperatura vs. Viabilidad')
sns.boxplot(ax=axes[1], x=recursos_vis['df']['viabilidad_eolica'], y=recursos_vis['df']['viento_promedio_anual_ms'], palette='viridis'); axes[1].set_title('Eólica: Viento vs. Viabilidad')
sns.boxplot(ax=axes[2], x=recursos_vis['df']['viabilidad_hidrica'], y=recursos_vis['df']['potencial_hidrico_proxy_mm'], palette='Blues'); axes[2].set_title('Hídrica: Precipitación vs. Viabilidad')
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig); plt.close(fig)

# Sub-card para la Importancia de Variables
st.markdown('<h3 style="color: #495057;">Figuras 3, 4 y 5: Importancia de Variables por Modelo</h3>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 0.9em; color: #6c757d;">*Figuras 3, 4 y 5. Gráficos de barras que muestran las 10 variables más influyentes para los modelos Solar, Eólico e Hídrico, respectivamente. Fuente: Elaboración propia.*</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    fig_solar = plot_importancia('modelo_regresion_solar.pkl', 'Solar', 'rocket')
    if fig_solar: st.pyplot(fig_solar); plt.close(fig_solar)
with col2:
    fig_eolica = plot_importancia('modelo_regresion_eolica.pkl', 'Eólica', 'mako')
    if fig_eolica: st.pyplot(fig_eolica); plt.close(fig_eolica)
with col3:
    fig_hidrica = plot_importancia('modelo_regresion_hidrica.pkl', 'Hídrica', 'cubehelix')
    if fig_hidrica: st.pyplot(fig_hidrica); plt.close(fig_hidrica)

# --- CARD 6: DETALLES TÉCNICOS (PLEGABLE) ---
with st.expander("🔬 Detalles Técnicos y Metodología"):
    st.markdown("""
    **Fuentes de Datos:**
    - **Climáticos:** Datos diarios de radiación solar, velocidad del viento, temperatura y precipitación obtenidos de la API NASA POWER para el período 2019-2023.
    - **Geográficos:** Altitud de las capitales provinciales y límites geográficos de las provincias.
    **Modelos de Machine Learning:**
    - Se entrenaron y compararon múltiples algoritmos, incluyendo Regresión Lineal, Random Forest y Gradient Boosting.
    - Los modelos seleccionados para esta aplicación son **Random Forest**, por su robustez y alto rendimiento.
    """)

# Añade este bloque justo antes del expander de "Detalles Técnicos"

# --- SECCIÓN 7: PRUEBAS DE APLICABILIDAD (SIMULADOR DE ESCENARIOS) ---
st.header("🧪 Pruebas de Aplicabilidad y Escenarios Hipotéticos")
st.markdown("""
Utilice el panel de control de la izquierda para definir las características de una ubicación hipotética y observe cómo el modelo responde en tiempo real.
""")

# --- CONTROLES PARA EL SIMULADOR EN LA BARRA LATERAL ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Simulador de Escenarios")

# Selector de energía para el simulador
energia_simulador = st.sidebar.selectbox(
    "Energía para Simulación:",
    ["Solar", "Eólica", "Hídrica"],
    key="simulador_energia"
)

# Inputs para las variables del modelo
st.sidebar.markdown("**Defina las Condiciones Hipotéticas:**")
sim_altitud = st.sidebar.number_input("Altitud (m.s.n.m.)", value=2850, min_value=0, max_value=6000, step=100)
sim_temp = st.sidebar.number_input("Temperatura Promedio (°C)", value=14.5, min_value=-5.0, max_value=35.0, step=0.5)
sim_viento = st.sidebar.number_input("Velocidad del Viento (m/s)", value=3.2, min_value=0.0, max_value=15.0, step=0.1)
sim_precip = st.sidebar.number_input("Precipitación Total Anual (mm)", value=1200.0, min_value=0.0, max_value=5000.0, step=50.0)

def llamar_api_simulacion(energia, altitud, temp, viento, precip):
    """
    Realiza una petición POST al nuevo endpoint de simulación de la API de FastAPI.
    """
    api_url = "http://127.0.0.1:8000/predecir_simulacion"
    payload = {
        "energia": energia,
        "altitud": altitud,
        "temp_promedio_anual_C": temp,
        "viento_promedio_anual_ms": viento,
        "potencial_hidrico_proxy_mm": precip
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de simulación: {e}")
        st.warning("Asegúrate de que el servidor de FastAPI esté corriendo.")
        return None 

# Botón para ejecutar la simulación
if st.sidebar.button("Ejecutar Simulación"):
    # Crear un DataFrame con los datos hipotéticos
    datos_simulados_dict = {
        "provincia": "Ubicación Hipotética",
        "altitud": sim_altitud,
        "temp_promedio_anual_C": sim_temp,
        "viento_promedio_anual_ms": sim_viento,
        "potencial_hidrico_proxy_mm": sim_precip
    }
    X_sim = pd.DataFrame([datos_simulados_dict])
    
    # Llamar a la API con los datos simulados
    resultado_simulacion = llamar_api_simulacion(energia_simulador, sim_altitud, sim_temp, sim_viento, sim_precip)
    
    # Mostrar los resultados de la simulación en una tarjeta destacada
    st.markdown("---")
    st.subheader("🎯 Resultado de la Simulación")
    st.markdown(f"<h3 style='text-align: center;'>Predicción para Energía {energia_simulador}</h3>", unsafe_allow_html=True)
    
    if resultado_simulacion:
        col1, col2, col3 = st.columns(3)
        pred_energia = resultado_simulacion['energia_anual_predicha']['valor']
        unidad_energia = resultado_simulacion['energia_anual_predicha']['unidad']
        pred_viabilidad = resultado_simulacion['viabilidad_predicha']

        col1.metric(f"Energía Anual ({unidad_energia})", f"{pred_energia:,.2f}")
        if 'potencia_media_predicha' in resultado_simulacion:
            potencia_kw = resultado_simulacion['potencia_media_predicha']['valor']
            col2.metric("Potencia Media (kW)", f"{potencia_kw:,.2f}")
        col3.metric("Viabilidad", pred_viabilidad)
        
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #0d6efd;">Interpretación del Escenario</h4>
            <p>Se ha evaluado una ubicación con las siguientes características: altitud de <strong>{sim_altitud} m</strong>, 
            temperatura de <strong>{sim_temp} °C</strong>, velocidad del viento de <strong>{sim_viento} m/s</strong> y precipitación de <strong>{sim_precip} mm</strong>.</p>
            <p>El modelo predice que este sitio tendría una viabilidad <strong>{pred_viabilidad.lower()}</strong> para la generación de energía {energia_simulador.lower()}, 
            con una producción anual estimada de <strong>{pred_energia:,.2f} {unidad_energia}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No se pudo obtener la predicción de la simulación.")


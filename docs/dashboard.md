# Dashboard — Streamlit

**Archivo:** `src/app/streamlit_app.py`
**Ejecutar:** `streamlit run src/app/streamlit_app.py`

## Funcionalidad

Dashboard interactivo para visualizar y predecir el potencial energético renovable en Ecuador.

## Layout

### Columna izquierda — Mapa interactivo (Folium)
- Mapa de Ecuador con las provincias coloreadas
- Tooltip muestra el nombre de cada provincia
- Click en cualquier punto del mapa selecciona la provincia automáticamente

### Columna derecha — Resultados de predicción
- Muestra 3 métricas en tarjetas:
  - ☀️ Potencial Solar (kWh/m²)
  - 💨 Potencial Eólico (MWh)
  - 💧 Potencial Hídrico (mm)

### Sidebar — Controles
- **Provincia seleccionada** (se actualiza automáticamente)
- **Slider: Mes del año** (1-12)
- **Slider: Temperatura promedio** (-10 a 40 °C)
- **Slider: Velocidad del viento** (0-20 m/s)
- **Slider: Precipitación total** (0-1000 mm)
- **Slider: Humedad relativa** (0-100%)
- **Botón "Predecir Potencial"** — llama a la API
- **Botón "Generar Reporte PDF"** — descarga reporte en PDF

## Dependencias externas
- La API (`src/api/main.py`) debe estar corriendo en `http://127.0.0.1:8000`

## Flujo de predicción
1. Usuario hace clic en el mapa → se guarda la provincia en `st.session_state`
2. Usuario ajusta sliders y presiona "Predecir"
3. Se calcula el centroide de la provincia seleccionada
4. Se envía `POST /predict` a la API con lat, lon, y parámetros climáticos
5. Los resultados se guardan en `st.session_state` y se muestran en pantalla

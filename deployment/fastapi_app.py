# deployment/fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import unicodedata
import os

# --- INICIALIZACIÓN DE LA APLICACIÓN FASTAPI ---
app = FastAPI(
    title="API de Predicción de Energías Renovables - Ecuador",
    description="API para predecir el potencial de generación de energía solar, eólica e hídrica en las provincias de Ecuador.",
    version="1.0.0"
)

# --- MODELOS PYDANTIC PARA VALIDACIÓN DE DATOS DE ENTRADA ---
# Esto asegura que la API reciba los datos en el formato correcto.
class DatosEntrada(BaseModel):
    provincia: str
    altitud: float
    temp_promedio_anual_C: float
    viento_promedio_anual_ms: float
    potencial_hidrico_proxy_mm: float

    class Config:
        schema_extra = {
            "example": {
                "provincia": "Pichincha",
                "altitud": 2850,
                "temp_promedio_anual_C": 14.5,
                "viento_promedio_anual_ms": 3.2,
                "potencial_hidrico_proxy_mm": 1200.0
            }
        }

# --- CARGA DE MODELOS Y RECURSOS (SE EJECUTA UNA SOLA VEZ AL INICIAR) ---
# Esto es eficiente, ya que los modelos no se recargan en cada petición.
print("Cargando modelos y recursos...")

# Diccionario para almacenar los modelos y codificadores
modelos = {}
try:
    # Modelos de Regresión
    modelos['reg_solar'] = joblib.load('modelo_regresion_solar.pkl')
    modelos['reg_eolica'] = joblib.load('modelo_regresion_eolica.pkl')
    modelos['reg_hidrica'] = joblib.load('modelo_regresion_hidrica.pkl')
    # Modelos de Clasificación
    modelos['clf_solar'] = joblib.load('modelo_clasificacion_solar.pkl')
    modelos['clf_eolica'] = joblib.load('modelo_clasificacion_eolica.pkl')
    modelos['clf_hidrica'] = joblib.load('modelo_clasificacion_hidrica.pkl')
    # Codificadores
    modelos['le_solar'] = joblib.load('label_encoder_solar.pkl')
    modelos['le_eolica'] = joblib.load('label_encoder_eolica.pkl')
    modelos['le_hidrica'] = joblib.load('label_encoder_hidrica.pkl')
    print("✅ Modelos y codificadores cargados exitosamente.")
except FileNotFoundError as e:
    print(f"❌ Error crítico: No se pudo encontrar un archivo de modelo. Asegúrate de que todos los archivos .pkl están en la carpeta 'deployment'. Detalles: {e}")
    # Opcional: Detener la app si no se cargan los modelos
    # import sys
    # sys.exit(1)

# --- FUNCIONES AUXILIARES ---
def normalizar_nombre(nombre):
    return unicodedata.normalize('NFKD', nombre).encode('ascii', errors='ignore').decode('utf-8')

def preparar_datos_para_prediccion(datos: DatosEntrada):
    """Convierte los datos de entrada en un DataFrame listo para el modelo."""
    datos_dict = datos.dict()
    df = pd.DataFrame([datos_dict])
    return df

# --- ENDPOINTS DE LA API ---

@app.get("/", tags=["General"])
def read_root():
    """
    Endpoint de bienvenida. Proporciona información sobre la API.
    """
    return {
        "message": "Bienvenido a la API de Predicción de Energías Renovables de Ecuador.",
        "version": "1.0.0",
        "usage": "Usa el endpoint /predecir con una petición POST para obtener predicciones. Ejemplo: /predecir?energia=Solar"
    }

@app.post("/predecir", tags=["Predicción"])
def predecir_potencial(
    datos: DatosEntrada,
    energia: str # Parámetro de consulta: ?energia=Solar
):
    """
    Realiza una predicción de energía y viabilidad para una provincia y tipo de energía dados.

    - **datos (body):** Un objeto JSON con las características geográficas y climáticas de la provincia.
    - **energia (query):** El tipo de energía a predecir. Valores posibles: 'Solar', 'Eolica', 'Hidrica'.
    """
    energia_capitalizada = energia.capitalize()
    
    # Seleccionar los modelos y codificador correctos
    if energia_capitalizada == 'Solar':
        modelo_reg = modelos.get('reg_solar')
        modelo_clf = modelos.get('clf_solar')
        le = modelos.get('le_solar')
        unidad = "kWh"
    elif energia_capitalizada == 'Eolica':
        modelo_reg = modelos.get('reg_eolica')
        modelo_clf = modelos.get('clf_eolica')
        le = modelos.get('le_eolica')
        unidad = "kWh"
    elif energia_capitalizada == 'Hidrica':
        modelo_reg = modelos.get('reg_hidrica')
        modelo_clf = modelos.get('clf_hidrica')
        le = modelos.get('le_hidrica')
        unidad = "mm"
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de energía no válido. Debe ser 'Solar', 'Eolica' o 'Hidrica'."
        )

    if not all([modelo_reg, modelo_clf, le]):
        raise HTTPException(
            status_code=500,
            detail=f"Los modelos para la energía '{energia_capitalizada}' no se cargaron correctamente."
        )

    # Preparar los datos para la predicción
    X_pred = preparar_datos_para_prediccion(datos)
    
    # Realizar predicciones
    prediccion_energia = modelo_reg.predict(X_pred)[0]
    prediccion_viabilidad_cod = modelo_clf.predict(X_pred)[0]
    prediccion_viabilidad = le.inverse_transform([prediccion_viabilidad_cod])[0]

    # Calcular potencia si aplica
    resultado = {
        "provincia": datos.provincia,
        "tipo_energia": energia_capitalizada,
        "energia_anual_predicha": {
            "valor": prediccion_energia,
            "unidad": unidad
        },
        "viabilidad_predicha": prediccion_viabilidad
    }
    
    if energia_capitalizada != 'Hidrica':
        resultado["potencia_media_predicha"] = {
            "valor": prediccion_energia / 365 / 24,
            "unidad": "kW"
        }

    return resultado


# --- CARGA DE DATOS PARA EL NUEVO ENDPOINT ---
# Cargamos el dataset una sola vez para que el endpoint pueda buscar los datos
try:
    df_datos = pd.read_csv('dataset_para_app_completo.csv')
    print("✅ Dataset para el endpoint /predecir_por_provincia cargado.")
except FileNotFoundError:
    print("❌ Error: No se encontró 'dataset_para_app_completo.csv'. El nuevo endpoint no funcionará.")
    df_datos = pd.DataFrame() # Crear un DataFrame vacío para evitar errores

# --- NUEVO ENDPOINT SIMPLIFICADO ---
# Reemplaza la función existente en deployment/fastapi_app.py por esta:

@app.get("/predecir_por_provincia", tags=["Predicción Simplificada"])
def predecir_por_provincia(
    provincia: str, 
    energia: str
):
    """
    Endpoint simplificado que predice el potencial solo con el nombre de la provincia y el tipo de energía.
    Busca los datos climáticos y geográficos internamente.
    """
    if df_datos.empty:
        raise HTTPException(status_code=500, detail="El dataset de datos no está disponible en el servidor.")

    # --- CORRECCIÓN: Normalizar el tipo de energía para eliminar tildes ---
    energia_normalizada = normalizar_nombre(energia)
    provincia_normalizada = normalizar_nombre(provincia)
    
    # Buscar los datos de la provincia en el dataset
    datos_provincia_df = df_datos[df_datos['provincia'].apply(normalizar_nombre) == provincia_normalizada]

    if datos_provincia_df.empty:
        raise HTTPException(status_code=404, detail=f"La provincia '{provincia}' no fue encontrada en los datos.")

    # Tomar la primera fila
    datos_provincia = datos_provincia_df.iloc[0]

    # Crear el objeto de entrada para la lógica de predicción existente
    datos_entrada = DatosEntrada(
        provincia=datos_provincia['provincia'],
        altitud=datos_provincia['altitud'],
        temp_promedio_anual_C=datos_provincia['temp_promedio_anual_C'],
        viento_promedio_anual_ms=datos_provincia['viento_promedio_anual_ms'],
        potencial_hidrico_proxy_mm=datos_provincia['potencial_hidrico_proxy_mm']
    )
    
    # --- CORRECCIÓN: Usar la versión normalizada para la comparación ---
    if energia_normalizada == 'Solar':
        modelo_reg = modelos.get('reg_solar')
        modelo_clf = modelos.get('clf_solar')
        le = modelos.get('le_solar')
        unidad = "kWh"
    elif energia_normalizada == 'Eolica': # Ahora 'Eólica' se convierte a 'Eolica'
        modelo_reg = modelos.get('reg_eolica')
        modelo_clf = modelos.get('clf_eolica')
        le = modelos.get('le_eolica')
        unidad = "kWh"
    elif energia_normalizada == 'Hidrica': # Ahora 'Hídrica' se convierte a 'Hidrica'
        modelo_reg = modelos.get('reg_hidrica')
        modelo_clf = modelos.get('clf_hidrica')
        le = modelos.get('le_hidrica')
        unidad = "mm"
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de energía no válido. Debe ser 'Solar', 'Eolica' o 'Hidrica'."
        )

    if not all([modelo_reg, modelo_clf, le]):
        raise HTTPException(
            status_code=500,
            detail=f"Los modelos para la energía '{energia}' no se cargaron correctamente."
        )

    # Preparar los datos para la predicción
    X_pred = preparar_datos_para_prediccion(datos_entrada)
    
    # Realizar predicciones
    prediccion_energia = modelo_reg.predict(X_pred)[0]
    prediccion_viabilidad_cod = modelo_clf.predict(X_pred)[0]
    prediccion_viabilidad = le.inverse_transform([prediccion_viabilidad_cod])[0]

    # Calcular potencia si aplica
    resultado = {
        "provincia": datos_entrada.provincia,
        "tipo_energia": energia.capitalize(), # Devolver el nombre original con tilde para la UI
        "energia_anual_predicha": {
            "valor": prediccion_energia,
            "unidad": unidad
        },
        "viabilidad_predicha": prediccion_viabilidad
    }
    
    if energia_normalizada != 'Hidrica':
        resultado["potencia_media_predicha"] = {
            "valor": prediccion_energia / 365 / 24,
            "unidad": "kW"
        }

    return resultado

# Añade este bloque al final de deployment/fastapi_app.py

# --- MODELO PYDANTIC PARA LA SIMULACIÓN ---
class DatosSimulacion(BaseModel):
    energia: str
    altitud: float
    temp_promedio_anual_C: float
    viento_promedio_anual_ms: float
    potencial_hidrico_proxy_mm: float

# --- NUEVO ENDPOINT PARA EL SIMULADOR ---
@app.post("/predecir_simulacion", tags=["Simulación"])
def predecir_simulacion_endpoint(datos: DatosSimulacion):
    """
    Realiza una predicción a partir de datos climáticos y geográficos proporcionados directamente,
    sin necesidad de buscar una provincia en un dataset. Ideal para escenarios hipotéticos.
    """
    energia_capitalizada = datos.energia.capitalize()
    
    # Seleccionar el modelo y codificador correctos
    if energia_capitalizada == 'Solar':
        modelo_reg = modelos.get('reg_solar')
        modelo_clf = modelos.get('clf_solar')
        le = modelos.get('le_solar')
        unidad = "kWh"
    elif energia_capitalizada == 'Eolica':
        modelo_reg = modelos.get('reg_eolica')
        modelo_clf = modelos.get('clf_eolica')
        le = modelos.get('le_eolica')
        unidad = "kWh"
    elif energia_capitalizada == 'Hidrica':
        modelo_reg = modelos.get('reg_hidrica')
        modelo_clf = modelos.get('clf_hidrica')
        le = modelos.get('le_hidrica')
        unidad = "mm"
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de energía no válido: {datos.energia}")

    if not all([modelo_reg, modelo_clf, le]):
        raise HTTPException(status_code=500, detail=f"Los modelos para la energía '{energia_capitalizada}' no se cargaron correctamente.")

    # Crear un DataFrame con los datos de la simulación
    # La columna 'provincia' es necesaria para el preprocesador, pero su valor no importa
    X_pred = pd.DataFrame([{
        'provincia': 'Simulacion', 
        'altitud': datos.altitud,
        'temp_promedio_anual_C': datos.temp_promedio_anual_C,
        'viento_promedio_anual_ms': datos.viento_promedio_anual_ms,
        'potencial_hidrico_proxy_mm': datos.potencial_hidrico_proxy_mm
    }])

    # Realizar predicciones
    prediccion_energia = modelo_reg.predict(X_pred)[0]
    prediccion_viabilidad_cod = modelo_clf.predict(X_pred)[0]
    prediccion_viabilidad = le.inverse_transform([prediccion_viabilidad_cod])[0]

    # Devolver el resultado
    resultado = {
        "tipo_energia": energia_capitalizada,
        "energia_anual_predicha": {"valor": prediccion_energia, "unidad": unidad},
        "viabilidad_predicha": prediccion_viabilidad
    }
    
    if energia_capitalizada != 'Hidrica':
        resultado["potencia_media_predicha"] = {"valor": prediccion_energia / 365 / 24, "unidad": "kW"}

    return resultado
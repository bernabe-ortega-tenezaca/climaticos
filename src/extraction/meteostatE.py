# Extractor de datos meteorológicos históricos desde Meteostat
from meteostat import Stations, Daily
from datetime import datetime
import pandas as pd


class MeteostatExtractor:

    @staticmethod
    def extraer_datos_ecuador(inicio: datetime = datetime(2020, 1, 1), fin: datetime = datetime(2024, 12, 31)) -> pd.DataFrame | None:
        """Descarga datos meteorológicos diarios de todas las estaciones en Ecuador.

        Args:
            inicio: Fecha de inicio del período de descarga.
            fin: Fecha de fin del período de descarga.

        Returns:
            DataFrame con datos meteorológicos concatenados de todas las estaciones,
            o None si no se encontraron estaciones.
        """
        stations = Stations().region('EC')
        estaciones_df = stations.fetch()
        print(f"Total estaciones: {len(estaciones_df)}")

        if estaciones_df.empty:
            return None

        lista_datos: list[pd.DataFrame] = []
        for estacion_id in estaciones_df.index:
            datos = Daily(estacion_id, inicio, fin).fetch()
            if not datos.empty:
                datos['Estacion'] = estacion_id
                lista_datos.append(datos)

        df_final = pd.concat(lista_datos)
        df_final.to_csv('./data/raw/meteostat_ecuador.csv')
        print("Datos guardados en data/raw/meteostat_ecuador.csv")
        return df_final


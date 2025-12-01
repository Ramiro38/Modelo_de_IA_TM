import os
# Esto obliga a TensorFlow a usar el modo compatibilidad con versiones antiguas
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from tensorflow import keras  
from PIL import Image, ImageOps  
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Reconocimiento Perros vs Gatos", page_icon="🐾")

st.title("🐶 Detector de Mascotas 🐱")
st.write("Usa la cámara para saber si es un perro o un gato.")

# DEFINIMOS UNA FUNCIÓN PARA CARGAR EL MODELO Y GUARDARLO EN CACHE
# Usamos cache para que no se cargue cada vez que detecta un movimiento
@st.cache_resource
def carga_modelo():
    # Cargamos el modelo
    modelo = keras.models.load_model("st-app/keras_model.h5", compile=False)
    # Carga las etiquetas de las clases
    clases = open("st-app/labels.txt", "r").readlines()
    return modelo, clases


# 1.CARGAMOS EL MODELO Y ETIQUETAS
try:
    mi_modelo, nombre_clases = carga_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

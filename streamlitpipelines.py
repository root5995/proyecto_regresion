import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predicción de Notas",
    page_icon="🎓",
    layout="wide"
)

# --- Título y descripción de la aplicación ---
st.title("🎓 Aplicación de Predicción de Notas de Examen")
st.markdown("---")
st.markdown("""
Esta aplicación utiliza un modelo de aprendizaje automático para predecir la nota de examen de un estudiante basándose en varias características. 
Ajusta los valores en el panel de la izquierda para ver cómo cambian las predicciones.
""")

# --- Carga del modelo (debe ser un pipeline de sklearn) ---
try:
    # Asegúrate de tener este archivo 'Modelopipeline.joblib' en el mismo directorio.
    model = load('Modelopipeline.joblib')
except FileNotFoundError:
    st.error("Error: El archivo 'Modelopipeline.joblib' no se encuentra. Por favor, asegúrate de que el modelo entrenado esté en el mismo directorio que este script.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --- Definición de las variables de entrada ---
# Utilizaremos Streamlit Session State para manejar el estado de las entradas,
# lo que nos permite crear un botón de "Resetear" efectivo.

if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'age': 20,
        'gender': 'Female',
        'study_hours_per_day': 3.0,
        'social_media_hours': 2.0,
        'netflix_hours': 1.0,
        'part_time_job': 'No',
        'attendance_percentage': 90.0,
        'sleep_hours': 7.0,
        'diet_quality': 'Good',
        'exercise_frequency': 3,
        'parental_education_level': 'High School',
        'internet_quality': 'Good',
        'mental_health_rating': 5,
        'extracurricular_participation': 'Yes'
    }

def reset_inputs():
    """Función para resetear las entradas a sus valores por defecto."""
    st.session_state.inputs = {
        'age': 20,
        'gender': 'Female',
        'study_hours_per_day': 3.0,
        'social_media_hours': 2.0,
        'netflix_hours': 1.0,
        'part_time_job': 'No',
        'attendance_percentage': 90.0,
        'sleep_hours': 7.0,
        'diet_quality': 'Good',
        'exercise_frequency': 3,
        'parental_education_level': 'High School',
        'internet_quality': 'Good',
        'mental_health_rating': 5,
        'extracurricular_participation': 'Yes'
    }

# --- Sidebar para la entrada del usuario ---
st.sidebar.header("📝 Entradas del Estudiante")
st.sidebar.markdown("Ajusta los parámetros para predecir la nota de examen.")

# Entradas para variables numéricas
st.session_state.inputs['age'] = st.sidebar.slider(
    "Edad", 18, 25, value=st.session_state.inputs['age'])

st.session_state.inputs['study_hours_per_day'] = st.sidebar.number_input(
    "Horas de estudio por día", min_value=0.0, max_value=24.0, 
    value=st.session_state.inputs['study_hours_per_day'], step=0.1)

st.session_state.inputs['social_media_hours'] = st.sidebar.number_input(
    "Horas de redes sociales", min_value=0.0, max_value=10.0, 
    value=st.session_state.inputs['social_media_hours'], step=0.1)

st.session_state.inputs['netflix_hours'] = st.sidebar.number_input(
    "Horas de Netflix", min_value=0.0, max_value=10.0, 
    value=st.session_state.inputs['netflix_hours'], step=0.1)

st.session_state.inputs['attendance_percentage'] = st.sidebar.number_input(
    "Porcentaje de Asistencia (%)", min_value = 0.0, max_value = 100.0, 
    value=st.session_state.inputs['attendance_percentage'], step=0.1)

st.session_state.inputs['sleep_hours'] = st.sidebar.number_input(
    "Horas de sueño", min_value=0.0, max_value=12.0, 
    value=st.session_state.inputs['sleep_hours'], step=0.1)

st.session_state.inputs['exercise_frequency'] = st.sidebar.slider(
    "Frecuencia de ejercicio (veces/semana)", 0, 7, 
    value=st.session_state.inputs['exercise_frequency'])

st.session_state.inputs['mental_health_rating'] = st.sidebar.slider(
    "Estado de salud mental (1-10)", 1, 10, 
    value=st.session_state.inputs['mental_health_rating'])

# Entradas para variables categóricas
st.session_state.inputs['gender'] = st.sidebar.selectbox(
    "Género", options=["Female", "Male", "Other"], 
    index=["Female", "Male", "Other"].index(st.session_state.inputs['gender']))

st.session_state.inputs['part_time_job'] = st.sidebar.selectbox(
    "Trabajo a tiempo parcial", options=["Yes", "No"], 
    index=["Yes", "No"].index(st.session_state.inputs['part_time_job']))

st.session_state.inputs['diet_quality'] = st.sidebar.selectbox(
    "Calidad de dieta", options=["Good", "Fair", "Poor"], 
    index=["Good", "Fair", "Poor"].index(st.session_state.inputs['diet_quality']))

st.session_state.inputs['parental_education_level'] = st.sidebar.selectbox(
    "Nivel de educación de los padres", options=["High School", "Master", "Bachelor", "PhD", "Associate"],
    index=["High School", "Master", "Bachelor", "PhD", "Associate"].index(st.session_state.inputs['parental_education_level']))

st.session_state.inputs['internet_quality'] = st.sidebar.selectbox(
    "Calidad de internet", options=["Good", "Average", "Poor"],
    index=["Good", "Average", "Poor"].index(st.session_state.inputs['internet_quality']))

st.session_state.inputs['extracurricular_participation'] = st.sidebar.selectbox(
    "Participación en extraescolares", options=["Yes", "No"],
    index=["Yes", "No"].index(st.session_state.inputs['extracurricular_participation']))


# Botones para acciones
if st.sidebar.button("Predecir Nota", type="primary"):
    # Crear un DataFrame con los datos de entrada
    obs = pd.DataFrame([st.session_state.inputs])

    # Mostrar el DataFrame de entradas para depuración
    st.subheader("Datos de entrada")
    st.write(obs)

    # Realizar la predicción
    try:
        predicted_score = model.predict(obs)[0]
        st.markdown("---")
        
        # Mostrar la predicción con un formato claro
        st.markdown(
            f'<div style="background-color: #d1e7dd; padding: 20px; border-radius: 10px; text-align: center; color: #0a3622;">'
            f'<h3>La nota de examen predicha es:</h3>'
            f'<h1 style="font-size: 60px; font-weight: bold; margin: 0; padding: 0;">{predicted_score:.2f}</h1>'
            f'</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

if st.sidebar.button("Resetear Valores"):
    reset_inputs()
    st.rerun()

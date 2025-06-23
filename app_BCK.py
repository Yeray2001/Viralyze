# streamlit
import streamlit as st
import pandas as pd
import joblib
from utils import preprocesar_X, puntuar_caption_ejes
import os
from dotenv import load_dotenv
import openai
import emoji
from textstat import flesch_reading_ease
from xgboost import XGBClassifier
import datetime
import locale
from PIL import Image


# Configuración general
st.set_page_config(page_title="Viralyze", page_icon="📈", layout="centered")

# Tipografía y fondo personalizados
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #E4D3FF 0%, #E4D3FF 25%, #FFC7BC 60%, #B8EFE7 100%);
        background-attachment: fixed;
    }

    html, body {
        font-family: 'Poppins', sans-serif;
        color: #0C1C38;
    }

    div.stButton > button {
        background-color: #B8EFE7;
        color: #0C1C38;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #A0E0D6;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Cargar clave API
load_dotenv("api.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Idioma para nombres de días
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES')
    except:
        st.warning("⚠️ No se pudo establecer el idioma local a español.")

# Logo
LOGO_PATH = "logo4.png"
try:
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH)
        resized_logo = logo.resize((150, 150))
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.image(resized_logo)
except Exception as e:
    st.warning(f"⚠️ Error al mostrar el logo: {e}")

# Cabecera
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#0C1C38; font-size: 40px;'>Viralyze</h1>
        <h4 style='color:#E24E94;'>Crea. Predice. Viraliza.</h4>
        <hr style='border: none; height: 3px; background: linear-gradient(to right, #9B4DFF, #FF6B5C);'>
    </div>
""", unsafe_allow_html=True)

# Modelo y columnas
modelo = joblib.load("modelo_xgb_calibrado.json")
columnas_modelo = joblib.load("columnas_modelo.pkl")
#st.write("🧪 Columnas modelo:", columnas_modelo)
umbral = 0.4

def sugerir_caption_mejorado(caption_original):
    prompt = f"""
Mejora este caption para Instagram para maximizar su claridad, originalidad, emoción, interacción, viralidad y valor para la audiencia. Tienes que enfocarte sobre todo en 
disparar el valor, claridad y originalidad. También debes añadir al menos 15 hashtags relevantes, etiquetar a tres cuentas de contenido beauty españa y aumentar la longitud del texto significativamente. Hazlo en un tono natural.

Caption original:
\"{caption_original}\"

Devuelve solo el caption mejorado, sin explicaciones.
"""
    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return respuesta.choices[0].message.content.strip()
    except:
        return ""


# Inputs
st.write("Completa los datos para evaluar la probabilidad de que tu publicación se vuelva viral.")
fecha = st.date_input("📅 Fecha de publicación", value=datetime.date.today())

# modificación aquí: forzar inglés para los días
try:
    locale.setlocale(locale.LC_TIME, "C")  # Establece formato en inglés
except:
    st.warning("⚠️ No se pudo establecer el idioma en inglés.")

dia = fecha.strftime("%A")  # Monday, Tuesday, etc.

hora = st.selectbox("🕒 Hora", list(range(0, 24)))
tipo = st.selectbox("🖼 Tipo de publicación", ["foto", "reel", "carrusel"])
caption = st.text_area("✍️ Escribe tu caption")


# Acción
if st.button("Predecir viralidad"):
    if not caption.strip():
        st.warning("Por favor, escribe un caption antes de predecir.")
    else:
        # Validar caption antes de predecir
        if len(caption.strip()) < 10 or len(caption.split()) < 2:
            st.error("❌ Este caption es demasiado corto o no tiene sentido. Intenta escribir algo más elaborado.")
            st.stop()

        # Obtener puntajes IA
        puntajes = puntuar_caption_ejes(caption)

        # Construir base del DataFrame
        df_usuario = pd.DataFrame({
            "is_night_post": [int(hora < 8 or hora >= 20)],
            "is_special_date": [0],
            "num_hashtags": [caption.count("#")],
            "num_mentions": [caption.count("@")],
            "num_emojis": [sum(char in emoji.EMOJI_DATA for char in caption)],
            "has_question": [int("?" in caption)],
            "has_call_to_action": [int(any(kw in caption.lower() for kw in [
                "sígueme", "comenta", "etiqueta", "dale like", "compártelo"
            ]))],
            "num_exclamations": [caption.count("!")],
            "readability": [flesch_reading_ease(caption) if caption else 50],
            "hashtag_ratio": [caption.count("#") / (len(caption.split()) + 1e-6)],
            "caption_length": [len(caption)],
        })

        # Agregar puntajes de claridad, emoción, etc.
        for k, v in puntajes.items():
            df_usuario[k] = v

        # One-hot para hora
        for h in range(24):
            df_usuario[f"hour_of_day_{h}"] = int(h == hora)

        # One-hot para día de la semana
        for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            df_usuario[f"day_of_week_{d}"] = int(d == dia)

        # One-hot para tipo de publicación
        for tipo_val in ["reel", "carrusel", "foto"]:
            df_usuario[f"media_type_{tipo_val}"] = int(tipo == tipo_val)

        # Asegurar que todas las columnas necesarias están
        for col in columnas_modelo:
            if col not in df_usuario.columns:
                df_usuario[col] = 0
        # Validación: asegurar coincidencia de columnas
        columnas_usuario = set(df_usuario.columns)
        columnas_esperadas = set(columnas_modelo)

        faltantes = columnas_esperadas - columnas_usuario
        sobrantes = columnas_usuario - columnas_esperadas

        if faltantes:
            st.error(f"❌ Faltan columnas necesarias para el modelo:\n{sorted(faltantes)}")

        if sobrantes:
            st.warning(f"⚠️ Estas columnas no se esperan por el modelo y serán ignoradas:\n{sorted(sobrantes)}")

        # Mostrar total y diferencias (útil para debugging)
        #st.caption(f"🔎 Total columnas esperadas por el modelo: {len(columnas_modelo)}")
        #st.caption(f"📄 Total columnas generadas por el usuario: {len(df_usuario.columns)}")
            
        # Asegurar orden correcto
        for col in columnas_modelo:
            if col not in df_usuario.columns:
                df_usuario[col] = 0

        X_usuario = df_usuario[columnas_modelo]


        prob = modelo.predict_proba(X_usuario)[0, 1] * 100
        es_viral = prob >= (umbral * 100)

        # Mostrar resultado
        st.subheader("Resultado")

        # Círculo visual con color dinámico
        color = "#FFB37B"  # Naranja coral
        if prob >= 66:
            color = "#A8E6CF"  # Verde pastel
        elif prob >= 33:
            color = "#FFEFA0"  # Amarillo pastel

        st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-top: 2em;">
                <div style="
                    position: relative;
                    width: 160px;
                    height: 160px;
                    border-radius: 50%;
                    background: conic-gradient({color} {prob}%, #e0e0e0 {prob}%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                ">
                    <div style="
                        position: absolute;
                        width: 120px;
                        height: 120px;
                        border-radius: 50%;
                        background-color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 28px;
                        font-weight: bold;
                        color: #0C1C38;
                    ">
                        {prob:.0f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if es_viral:
            st.success("🌟 Este post tiene altas probabilidades de ser viral")
        else:
            st.warning("📉 Este post no parece muy viral aún")

        # Mostrar entrada procesada
        #with st.expander("🔍 Ver variables utilizadas por el modelo"):
            #st.dataframe(X_usuario)

        st.subheader("Recomendaciones")
        
        # Reglas basadas en interpretación SHAP
        if df_usuario["num_hashtags"].iloc[0] < 5:
            st.info("🔖 Usa un número elevado de hashtags para mejorar visibilidad.")
            
        if df_usuario["hashtag_ratio"].iloc[0] < 0.2:
            st.info("📊 Aumenta la proporción de hashtags respecto al texto.")

        if df_usuario["has_question"].iloc[0] == 0:
            st.info("❓ Incluir un par de preguntas puede incrementar la interacción.")

        if df_usuario["num_emojis"].iloc[0] < 1:
            st.info("😄 Deberías incluir algún emoji para hacerlo más visual. No te pases o conseguirás el efecto contrario.")

        if df_usuario["day_of_week_Thursday"].iloc[0] == 1 or df_usuario["day_of_week_Monday"].iloc[0] == 1:
            st.info("📅 Los jueves y los lunes no son el mejor día para publicar. Considera publicar otro día.")

        if df_usuario["hour_of_day_19"].iloc[0] == 1 or df_usuario["hour_of_day_12"].iloc[0] == 1:
            st.info("🕒 Publicar a las 12:00 y sobre todo 19:00 puede disminuir en gran medida la visibilidad. Prueba a publicar a las 11:00.")

        if df_usuario["num_mentions"].iloc[0] < 1:
            st.info("💡 Considera mencionar al menos una cuenta relacionada, eso puede aumentar la visibilidad de tu post.")
        

        media = sum(puntajes.values()) / len(puntajes)
        if media < 8:
            st.subheader("🤖 Caption sugerido por la IA")
            for intento in range(5):
                sugerido = sugerir_caption_mejorado(caption)
                if sugerido.strip():
                    st.text_area("Sugerencia automática:", sugerido, height=100)
                    break
            else:
                st.warning("⚠️ No se pudo generar una sugerencia válida.")

# Pie de página
st.markdown("""
<div style='text-align: center; color: gray; font-size: 13px; margin-top: 2em;'>
    Viralyze es un proyecto académico sin ánimo de lucro – 2025
</div>
""", unsafe_allow_html=True)

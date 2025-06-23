# al final sin uso, dejo el código por si puede ser útil

import pandas as pd
import random
import numpy as np
import time
from utils import puntuar_caption_ejes

# Cargar dataset original
df_original = pd.read_csv("instagram_caption_enriquecido.csv")
cuentas = df_original["account_name"].unique()

# Posibles valores aleatorios para variables necesarias
dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
tipos = ["foto", "reel", "carrusel"]
horas = list(range(0, 24))

sinteticos = []

print("🧪 Generando captions incoherentes...")
for cuenta in cuentas:
    for _ in range(5):
        texto_sin_sentido = "".join(random.choices("asdfghjklqwertyuiopzxcvbnm", k=random.randint(5, 15)))
        dia = random.choice(dias)
        hora = random.choice(horas)
        tipo = random.choice(tipos)

        intentos = 0
        puntajes = None
        while intentos < 3:
            puntajes = puntuar_caption_ejes(texto_sin_sentido)
            if puntajes and all(v != 5 for v in puntajes.values()):
                break
            print("⏳ Reintentando...")
            time.sleep(5)
            intentos += 1

        if not puntajes:
            print("❌ No se pudo puntuar:", texto_sin_sentido)
            continue

        sinteticos.append({
            "account_name": cuenta,
            "caption": texto_sin_sentido,
            "caption_length": len(texto_sin_sentido),
            "is_night_post": int(hora < 8 or hora >= 20),
            "is_special_date": 0,
            "num_hashtags": texto_sin_sentido.count("#"),
            "num_mentions": texto_sin_sentido.count("@"),
            "num_emojis": 0,
            "has_question": int("?" in texto_sin_sentido),
            "has_call_to_action": 0,
            "hashtag_ratio": 0,
            "readability": 0,
            "num_exclamations": texto_sin_sentido.count("!"),
            "media_type": tipo,
            "day_of_week": dia,
            "hour_of_day": hora,
            "followers": 1000,
            "followees": 500,
            "engagement_ratio": 0.0001,
            "comment_like_ratio": 0.0,
            **puntajes
        })

# Convertir a DataFrame y guardar
df_sinteticos = pd.DataFrame(sinteticos)
df_completo = pd.concat([df_original, df_sinteticos], ignore_index=True)
df_completo.to_csv("instagram_caption_enriquecido_expandido.csv", index=False)

print("✅ Dataset extendido guardado como instagram_caption_enriquecido_expandido.csv")

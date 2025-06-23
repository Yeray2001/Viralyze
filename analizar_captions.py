# analizar_captions.py: idea de Julio. Llamar a la IA para puntuar captions de Instagram en cuanto a 
# 6 ejes: claridad, originalidad, emocion, interaccion, redaccion_viral y valor
# Probamos con la nueva BBDD para ver si mejora nuestro modelo, además de añadir recomendaciones a Streamlit.

import pandas as pd
import time
from utils import puntuar_caption_ejes
import dotenv
import os

# Cargar clave API desde api.env
dotenv.load_dotenv(dotenv_path="api.env")

# Cargar datos
lote1_path = "instagram_lote1.csv"
lote2_path = "instagram_lote2.csv"
df1 = pd.read_csv(lote1_path)
df2 = pd.read_csv(lote2_path)
df = pd.concat([df1, df2], ignore_index=True)  # Todo el dataset

# Evitar repetir análisis si ya existe
output_file = "instagram_caption_enriquecido.csv"
try:
    df_ejes_existente = pd.read_csv(output_file)
    captions_ya_analizados = df_ejes_existente.shape[0]
    print(f"🔁 Reanudando desde la fila {captions_ya_analizados}")
    resultados = df_ejes_existente.to_dict(orient="records")
except FileNotFoundError:
    resultados = []
    captions_ya_analizados = 0

# Iterar y puntuar uno a uno
print("🔍 Evaluando captions con IA...")
for i, caption in enumerate(df["caption"].fillna("")[captions_ya_analizados:], start=captions_ya_analizados):
    print(f"{i+1}/{len(df)}")
    puntajes = puntuar_caption_ejes(caption)
    resultados.append(puntajes)
    pd.DataFrame(resultados).to_csv(output_file, index=False)
    time.sleep(21)  # Adaptado al límite de 3 RPM para cuentas sin pago

# Guardado final
df_ejes = pd.DataFrame(resultados)
df_final = pd.concat([df, df_ejes], axis=1)
df_final.to_csv(output_file, index=False)
print(f"✅ Captions analizados y guardados en: {output_file}")

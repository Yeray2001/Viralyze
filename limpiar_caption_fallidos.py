# por si alguna llamada a la IA no ha funcionado

import pandas as pd

# Cargar el archivo enriquecido
ruta = "instagram_caption_enriquecido.csv"
df = pd.read_csv(ruta)

# Definir columnas de puntuación
ejes = ["claridad", "originalidad", "emocion", "interaccion", "redaccion_viral", "valor"]

# Detectar filas donde todos los ejes valen 5
fallidos = df[ejes].eq(5).all(axis=1)

# Informar
print(f"Encontradas {fallidos.sum()} observaciones con puntuaciones 5 en todos los ejes.")

# Eliminar los fallidos y guardar el nuevo CSV
df_filtrado = df[~fallidos].reset_index(drop=True)
df_filtrado[ejes] = df_filtrado[ejes].fillna(0)  # Rellenar NaN si hay columnas nuevas vacías
df_filtrado[ejes] = df_filtrado[ejes].astype(int)
df_filtrado.to_csv("instagram_caption_enriquecido.csv", index=False)

print("✅ Archivo limpiado guardado: instagram_caption_enriquecido.csv")

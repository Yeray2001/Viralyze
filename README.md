# Viralyze – Predicción de Viralidad en Instagram

**Viralyze** es una aplicación interactiva construida con **Streamlit** que permite predecir la probabilidad de que una publicación de Instagram se vuelva viral. Utiliza procesamiento de texto, análisis semántico con inteligencia artificial y un modelo de machine learning calibrado para ofrecer no solo una predicción, sino también **recomendaciones personalizadas** para mejorar el rendimiento de tus publicaciones.

---

## Objetivo del Proyecto

El objetivo es asistir a creadores de contenido, community managers y marcas a:

- Anticipar si una publicación será viral.
- Mejorar su contenido textual con sugerencias basadas en IA.
- Optimizar variables como el momento de publicación, número de hashtags o tipo de media.

---

## ¿Qué te pide Viralyze?
- Introducción de :
    - fecha exacta de publicación
    - hora 
    - tipo de contenido (foto/ carrusel /reel)
    - caption

## ¿Qué te ofrece?
- Predicción de viralidad mediante modelo XGBoost calibrado.
- Recomendaciones prácticas basadas en análisis SHAP.
- Generación automática de captions sugeridos mediante IA (OpenAI GPT).

---

## Modelo de Machine Learning

- **Algoritmo base**: `XGBoostClassifier`.
- **Calibración**: `CalibratedClassifierCV(method="sigmoid")` para ajustar las probabilidades.
- **Balanceo de clases**: aplicado `SMOTE` sobre el conjunto de entrenamiento.
- **Umbral de clasificación**: `0.4`, optimizado para mejorar el recall en clase positiva (posts virales).

### Rendimiento (con test set)

| Métrica         | Valor     |
|-----------------|-----------|
| Accuracy        | 92%       |
| Precision (viral) | 82%    |
| Recall (viral)  | 88%       |
| F1-score (viral)| 85%       |

---

## Interpretabilidad

- Se utilizó la importancia de variables de XGBoost para entender qué factores afectan más a la viralidad.
- Además, se calcularon valores **SHAP** para interpretar el **impacto individual** de cada variable en cada predicción.

---

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/viralyze.git
cd viralyze
```

### 2. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecuta la app

```bash
streamlit run app.py
```

---

## Créditos

Este proyecto fue desarrollado como parte del Proyecto Final del Máster en Dacta Science e IA.  
Autora: **Elena Millán**  

---

**Crea. Predice. Viraliza. – Viralyze**
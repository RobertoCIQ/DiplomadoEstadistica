# 🚗 Predicción de Precios de Automóviles

Aplicación interactiva de Machine Learning para predecir precios de automóviles usando múltiples algoritmos y técnicas de optimización.


## 📋 Requisitos

- Python 3.10 o superior
- Dependencias listadas en la sección de instalación

## 🔧 Instalación

1. **Clonar o descargar el proyecto**

2. **Instalar las dependencias necesarias:**

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly
```

Esto instalará:
- `streamlit` - Framework de la aplicación web
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Modelos de ML y herramientas de preprocesamiento
- `xgboost` - Modelo XGBoost
- `plotly` - Visualizaciones interactivas

## 🚀 Ejecución

Para ejecutar la aplicación Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📊 Estructura del Proyecto

```
TareaDiplomadoIslas/
├── streamlit_app.py              # Aplicación principal de Streamlit
├── alexis_data_challenge_.py     # Modelo base inicial 
├── analisis_exploratorio.py      # Análisis exploratorio de features
├── car_price_prediction.csv      # Dataset de entrenamiento
└── README.md                     # Este archivo
```

## 🎯 Mejor Configuración del Modelo

Después de realizar pruebas exhaustivas, la configuración óptima es:

### **Configuración Recomendada:**

- **Algoritmo:** Random Forest
- **PCA:** Sí (6 componentes)
- **Optimización de Categorías:** Sí
- **Profundidad Máxima:** None (sin límite)
- **Número de Estimadores:** 100 (default)

### **Características de esta configuración:**

✅ **PCA activado** reduce la dimensionalidad de variables numéricas a 6 componentes principales, capturando la mayor parte de la varianza

✅ **Optimización de categorías** agrupa fabricantes y modelos con baja frecuencia en categorías de gama (Alta/Media/Baja) según precio promedio

✅ **Random Forest sin límite de profundidad** permite al modelo capturar patrones complejos disminuyendo overfitting gracias al ensemble de árboles

✅ **Balance óptimo** entre precisión, generalización y tiempo de entrenamiento

## 🔍 Funcionalidades

### **1. Configuración del Modelo**
- Selección entre 4 algoritmos: Decision Tree, Random Forest, XGBoost, Linear Regression
- Activación/desactivación de PCA
- Optimización automática de variables categóricas
- Ajuste de hiperparámetros

### **2. Métricas de Evaluación**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)
- R² Ajustado
- Varianza explicada (con PCA)

### **3. Visualizaciones**
- Scatter Plot: Predicciones vs Valores Reales
- Gráfico de Residuos
- Estadísticas de errores

### **4. Análisis de Features**
- Importancia de variables (para modelos basados en árboles)
- Componentes principales (cuando PCA está activo)

## 📈 Pipeline de Datos

1. **Carga y Limpieza:**
   - Extracción de valores numéricos de columnas mixtas
   - Eliminación de outliers (precio > media + 3σ)
   - Conversión de tipos de datos

2. **Ingeniería de Features:**
   - Agrupación de fabricantes con <50 registros por gama de precio
   - Agrupación de modelos con <15 registros por gama de precio
   - One-Hot Encoding de variables categóricas

3. **Reducción Dimensional (opcional):**
   - PCA sobre variables numéricas
   - Escalado con StandardScaler

4. **Entrenamiento y Evaluación:**
   - Split 80/20 (train/test)
   - Entrenamiento del modelo seleccionado
   - Cálculo de métricas de rendimiento
  
5. **Aplicacion:**

```bash
https://diplomadoestadistica.onrender.com/
```

## 📝 Notas Adicionales

- El dataset contiene información de 18,969 vehículos después de la limpieza
- Las variables categóricas de alta cardinalidad (Manufacturer: 63, Model: 1539) se optimizan automáticamente
- La aplicación usa caché de Streamlit para optimizar el rendimiento

## 👨‍💻 Uso

1. Ejecuta la aplicación
2. Configura los parámetros del modelo en la barra lateral
3. Haz clic en "Entrenar Modelo"
4. Revisa las métricas y visualizaciones generadas
5. Experimenta con diferentes configuraciones para comparar resultados

---


# -*- coding: utf-8 -*-
"""
Streamlit App Optimizada - Car Price Prediction con PCA, XGBoost y Agrupación de Categorías
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Precios de Autos - Optimizada",
    page_icon="🚗",
    layout="wide"
)

# Función para calcular R² ajustado
def adjusted_r2(r2, n, p):
    """Calcula R² ajustado dado R², número de muestras n y número de predictores p"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Función para agrupar manufacturer
def group_manufacturer(df):
    """Agrupa manufacturers con menos de 5 registros en categorías de gama por precio"""
    # Contar ocurrencias de cada manufacturer
    manufacturer_counts = df['Manufacturer'].value_counts()
    
    # Calcular precio promedio por manufacturer
    manufacturer_avg_price = df.groupby('Manufacturer')['Price'].mean()
    
    def categorize_manufacturer(row):
        manufacturer = row['Manufacturer']
        count = manufacturer_counts.get(manufacturer, 0)
        
        # Si tiene 50 o más registros, mantener el nombre original
        if count >= 50:
            return manufacturer
        
        # Si tiene menos de 50, agrupar por gama según precio promedio
        avg_price = manufacturer_avg_price.get(manufacturer, row['Price'])
        
        if avg_price >= 30000:
            return 'Otros - Gama Alta'
        elif avg_price >= 15000:
            return 'Otros - Gama Media'
        else:
            return 'Otros - Gama Baja'
    
    df['Manufacturer_Grouped'] = df.apply(categorize_manufacturer, axis=1)
    return df

# Función para agrupar modelos
def group_models(df):
    """Agrupa modelos con menos de 5 registros en categorías de gama por precio"""
    # Contar ocurrencias de cada modelo
    model_counts = df['Model'].value_counts()
    
    # Calcular precio promedio por modelo
    model_avg_price = df.groupby('Model')['Price'].mean()
    
    def categorize_model(row):
        model = row['Model']
        count = model_counts.get(model, 0)
        
        # Si tiene 15 o más registros, mantener el nombre original
        if count >= 15:
            return model
        
        # Si tiene menos de 15, agrupar por gama según precio promedio
        avg_price = model_avg_price.get(model, row['Price'])
        
        if avg_price >= 30000:
            return 'Otros Modelos - Gama Alta'
        elif avg_price >= 15000:
            return 'Otros Modelos - Gama Media'
        else:
            return 'Otros Modelos - Gama Baja'
    
    df['Model_Grouped'] = df.apply(categorize_model, axis=1)
    return df

# Función de carga y limpieza de datos
@st.cache_data
def load_and_clean_data(optimize_features=False):
    """Carga y limpia los datos del CSV"""
    df = pd.read_csv("car_price_prediction.csv")
    
    # Limpieza de columnas
    df["Engine volume"] = df["Engine volume"].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
    df["Engine volume"] = pd.to_numeric(df["Engine volume"], errors='coerce')
    
    df["Doors"] = df["Doors"].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
    df["Doors"] = pd.to_numeric(df["Doors"], errors='coerce')
    
    df["Mileage"] = df["Mileage"].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors='coerce')
    
    df['Levy'] = df['Levy'].replace('-', 0).astype(int)
    
    # Eliminar outliers principales
    max_price_index = df['Price'].idxmax()
    df = df.drop(max_price_index).reset_index(drop=True)
    
    max_engine_index = df['Engine volume'].idxmax()
    df = df.drop(max_engine_index).reset_index(drop=True)
    
    # Eliminar outliers de precio
    mean_price = df['Price'].mean()
    std_price = df['Price'].std()
    upper_bound = mean_price + (3 * std_price)
    df = df[df['Price'] < upper_bound]
    
    if optimize_features:
        df = group_manufacturer(df)
        df = group_models(df)
    
    return df

# Función para preparar datos y entrenar modelo
@st.cache_data
def prepare_and_train_model(model_type="Decision Tree", max_depth=None, n_estimators=100, 
                           use_pca=False, n_components=6, optimize_categories=False,
                           learning_rate=0.1):
    """Prepara datos y entrena el modelo seleccionado"""
    df = load_and_clean_data(optimize_features=optimize_categories)
    
    # Seleccionar columnas para encoding
    if optimize_categories:
        columns_to_dummy = ['Manufacturer_Grouped', 'Model_Grouped', 'Gear box type', "Category", 
                            'Leather interior', 'Fuel type', 'Drive wheels', 'Wheel']
    else:
        columns_to_dummy = ['Manufacturer', 'Model', 'Gear box type', "Category", 
                            'Leather interior', 'Fuel type', 'Drive wheels', 'Wheel']
    
    df_encoded = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)
    df_encoded = df_encoded.drop(columns=['Color'], errors='ignore')
    
    if optimize_categories:
        df_encoded = df_encoded.drop(columns=['Manufacturer', 'Model'], errors='ignore')
    
    # Separar features y target
    y = df_encoded['Price']
    features_to_exclude = ['ID', 'Price']
    X = df_encoded.drop(columns=features_to_exclude, errors='ignore')
    
    # Limpiar nombres de columnas para compatibilidad con XGBoost
    X.columns = X.columns.str.replace('[', '_', regex=False)
    X.columns = X.columns.str.replace(']', '_', regex=False)
    X.columns = X.columns.str.replace('<', '_', regex=False)
    X.columns = X.columns.str.replace('>', '_', regex=False)
    X.columns = X.columns.str.replace(',', '_', regex=False)
    
    # Aplicar PCA si se solicita
    scaler_obj = None
    pca_obj = None
    
    if use_pca:
        numeric_cols = ['Levy', 'Prod. year', 'Cylinders', 'Airbags', 
                       'Engine volume', 'Mileage', 'Doors']
        
        X_numeric = X[numeric_cols]
        X_categorical = X.drop(columns=numeric_cols)
        
        scaler_obj = StandardScaler()
        X_scaled = scaler_obj.fit_transform(X_numeric)
        
        pca_obj = PCA(n_components=n_components)
        X_pca = pca_obj.fit_transform(X_scaled)
        
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        X = pd.concat([X_pca_df, X_categorical], axis=1)
        variance_explained = sum(pca_obj.explained_variance_ratio_)
    else:
        variance_explained = None
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo según tipo
    if model_type == "Decision Tree":
        if max_depth:
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
    elif model_type == "Random Forest":
        if max_depth:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    elif model_type == "XGBoost":
        if max_depth:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
    else:  # Linear Regression
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "Num Features": X_train.shape[1],
        "Variance Explained (PCA)": variance_explained
    }
    
    return model, X, df, metrics, y_test, y_pred, X_train, scaler_obj, pca_obj

# Inicializar session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Título principal
st.title("🚗 Predicción de Precios de Automóviles")
st.markdown("### Modelo Predictivo con PCA, XGBoost y Optimización de Categorías")

# Crear tabs - Solo Predicción y Analítica
tab1, tab2 = st.tabs(["📊 Predicción", "📈 Analítica del Modelo"])

# ==================== TAB 1: PREDICCIÓN (TODO EN UNO) ====================
with tab1:
    st.header("Configuración y Predicción del Modelo")
    
    # ===== SECCIÓN 1: CONFIGURACIÓN DEL MODELO =====
    st.subheader("🔧 1. Configuración del Modelo")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        model_type = st.selectbox(
            "Algoritmo",
            ["XGBoost", "Random Forest", "Decision Tree", "Linear Regression"],
            index=0
        )
        
        if model_type in ["Decision Tree", "Random Forest", "XGBoost"]:
            use_max_depth = st.checkbox("Limitar Profundidad", value=False, key="depth_check")
            if use_max_depth:
                max_depth = st.slider("Profundidad Máxima", 1, 50, 10, key="max_depth_slider")
            else:
                max_depth = None
        else:
            max_depth = None
    
    with col_config2:
        if model_type in ["Random Forest", "XGBoost"]:
            n_estimators = st.slider("N° Estimadores", 10, 200, 100, step=10, key="n_est")
        else:
            n_estimators = 100
        
        if model_type == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key="lr")
        else:
            learning_rate = 0.1
    
    with col_config3:
        use_pca = st.checkbox("🔬 Usar PCA", value=False, key="pca_check")
        if use_pca:
            n_components = st.slider("Componentes PCA", 3, 7, 6, key="pca_comp")
        else:
            n_components = 6
        
        optimize_categories = st.checkbox("📦 Optimizar Categorías", value=False, key="opt_cat")
    
    # Botón de entrenar
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            model, X, df, metrics, y_test, y_pred, X_train, scaler_obj, pca_obj = prepare_and_train_model(
                model_type=model_type,
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                use_pca=use_pca,
                n_components=n_components,
                optimize_categories=optimize_categories
            )
            
            # Guardar en session state
            st.session_state.model_trained = True
            st.session_state.train_analytics = True
            st.session_state.model_config = {
                'model_type': model_type,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'use_pca': use_pca,
                'n_components': n_components,
                'optimize_categories': optimize_categories
            }
            st.session_state.model_metrics = metrics
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.trained_model = model
            st.session_state.X_template = X
            st.session_state.scaler = scaler_obj
            st.session_state.pca = pca_obj
            st.session_state.df_for_prediction = df
        
        st.success("✅ Modelo entrenado exitosamente!")
    
    # ===== SECCIÓN 2: CARACTERÍSTICAS DEL MODELO =====
    if st.session_state.model_trained:
        st.divider()
        st.subheader("📋 2. Características del Modelo Entrenado")
        
        config = st.session_state.model_config
        metrics = st.session_state.model_metrics
        X_template = st.session_state.X_template
        
        # Mostrar métricas del modelo
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("R²", f"{metrics['R2']:.4f}")
        with metric_cols[1]:
            st.metric("R² Ajustado", f"{metrics['Adjusted R2']:.4f}")
        with metric_cols[2]:
            st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
        with metric_cols[3]:
            st.metric("MAE", f"${metrics['MAE']:,.0f}")
        with metric_cols[4]:
            st.metric("Features", metrics['Num Features'])
        
        # Tabla de características resumida
        st.write("**Características utilizadas en el modelo:**")
        
        # Analizar características
        features_list = list(X_template.columns)
        
        # Agrupar por tipo de feature
        feature_summary = []
        
        # Contar dummies de Manufacturer
        manufacturer_features = [f for f in features_list if 'Manufacturer' in f or 'Manufacturer_Grouped' in f]
        if manufacturer_features:
            feature_summary.append({
                'Tipo de Variable': 'Manufacturer_Dummy',
                'Cantidad': len(manufacturer_features),
                'Descripción': 'Variables dummy para fabricantes de vehículos'
            })
        
        # Contar dummies de Model
        model_features = [f for f in features_list if 'Model' in f and 'Manufacturer' not in f]
        if model_features:
            feature_summary.append({
                'Tipo de Variable': 'Model_Dummy',
                'Cantidad': len(model_features),
                'Descripción': 'Variables dummy para modelos de vehículos'
            })
        
        # Componentes PCA si existen
        pca_features = [f for f in features_list if f.startswith('PC')]
        if pca_features:
            feature_summary.append({
                'Tipo de Variable': 'PCA_Components',
                'Cantidad': len(pca_features),
                'Descripción': 'Componentes principales (reducción dimensional)'
            })
        
        # Variables numéricas originales (si no hay PCA)
        numeric_originals = ['Levy', 'Prod. year', 'Cylinders', 'Airbags', 'Engine volume', 'Mileage', 'Doors']
        numeric_features = [f for f in features_list if f in numeric_originals]
        if numeric_features:
            feature_summary.append({
                'Tipo de Variable': 'Variables_Numéricas',
                'Cantidad': len(numeric_features),
                'Descripción': 'Variables numéricas originales'
            })
        
        # Otras variables categóricas
        other_categorical = [f for f in features_list if f not in manufacturer_features + model_features + pca_features + numeric_features]
        if other_categorical:
            # Agrupar por prefijo
            categorical_groups = {}
            for feat in other_categorical:
                prefix = feat.split('_')[0] if '_' in feat else feat
                if prefix not in categorical_groups:
                    categorical_groups[prefix] = []
                categorical_groups[prefix].append(feat)
            
            for prefix, feats in categorical_groups.items():
                feature_summary.append({
                    'Tipo de Variable': f'{prefix}_Dummy',
                    'Cantidad': len(feats),
                    'Descripción': f'Variables dummy para {prefix}'
                })
        
        # Mostrar tabla resumida
        summary_df = pd.DataFrame(feature_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Expander con detalles completos
        with st.expander("🔍 Ver detalles completos de todas las características"):
            st.write(f"**Total de características: {len(features_list)}**")
            
            # Organizar en tabs por tipo
            if manufacturer_features or model_features or other_categorical:
                detail_tabs = []
                if manufacturer_features:
                    detail_tabs.append("Manufacturer")
                if model_features:
                    detail_tabs.append("Model")
                if pca_features:
                    detail_tabs.append("PCA")
                if numeric_features:
                    detail_tabs.append("Numéricas")
                if other_categorical:
                    detail_tabs.append("Otras Categóricas")
                
                tabs = st.tabs(detail_tabs)
                
                tab_idx = 0
                if manufacturer_features:
                    with tabs[tab_idx]:
                        st.write(f"**{len(manufacturer_features)} variables:**")
                        for i, feat in enumerate(manufacturer_features, 1):
                            st.text(f"{i}. {feat}")
                    tab_idx += 1
                
                if model_features:
                    with tabs[tab_idx]:
                        st.write(f"**{len(model_features)} variables:**")
                        cols = st.columns(3)
                        for i, feat in enumerate(model_features):
                            with cols[i % 3]:
                                st.text(f"{i+1}. {feat}")
                    tab_idx += 1
                
                if pca_features:
                    with tabs[tab_idx]:
                        st.write(f"**{len(pca_features)} componentes:**")
                        for i, feat in enumerate(pca_features, 1):
                            st.text(f"{i}. {feat}")
                    tab_idx += 1
                
                if numeric_features:
                    with tabs[tab_idx]:
                        st.write(f"**{len(numeric_features)} variables:**")
                        for i, feat in enumerate(numeric_features, 1):
                            st.text(f"{i}. {feat}")
                    tab_idx += 1
                
                if other_categorical:
                    with tabs[tab_idx]:
                        st.write(f"**{len(other_categorical)} variables:**")
                        for i, feat in enumerate(other_categorical, 1):
                            st.text(f"{i}. {feat}")
        
        # ===== SECCIÓN 3: FORMULARIO DE PREDICCIÓN =====
        st.divider()
        st.subheader("🎯 3. Realizar Predicción")
        
        optimize_categories = config.get('optimize_categories', False)
        df_for_prediction = st.session_state.df_for_prediction
        
        # Columnas para inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if optimize_categories:
                manufacturer_options = sorted(df_for_prediction['Manufacturer_Grouped'].unique())
                manufacturer_grouped = st.selectbox("Categoría de Fabricante", manufacturer_options, key="manu_group")
                
                available_models = df_for_prediction[
                    df_for_prediction['Manufacturer_Grouped'] == manufacturer_grouped
                ]['Model_Grouped'].unique()
                model_grouped = st.selectbox("Categoría de Modelo", sorted(available_models), key="model_group")
            else:
                manufacturer = st.selectbox("Fabricante", sorted(df_for_prediction['Manufacturer'].unique()), key="manu")
                
                available_models = df_for_prediction[
                    df_for_prediction['Manufacturer'] == manufacturer
                ]['Model'].unique()
                model = st.selectbox("Modelo", sorted(available_models), key="model")
            
            category = st.selectbox("Categoría", sorted(df_for_prediction['Category'].unique()), key="cat")
            leather = st.selectbox("Interior de Cuero", sorted(df_for_prediction['Leather interior'].unique()), key="leather")
            fuel = st.selectbox("Tipo de Combustible", sorted(df_for_prediction['Fuel type'].unique()), key="fuel")
        
        with col2:
            gear = st.selectbox("Caja de Cambios", sorted(df_for_prediction['Gear box type'].unique()), key="gear")
            drive_wheels = st.selectbox("Tracción", sorted(df_for_prediction['Drive wheels'].unique()), key="drive")
            wheel = st.selectbox("Posición del Volante", sorted(df_for_prediction['Wheel'].unique()), key="wheel")
            engine_volume = st.number_input("Volumen Motor (L)", 0.0, 10.0, 2.0, 0.1, key="engine")
            mileage = st.number_input("Kilometraje", 0, 1000000, 50000, 1000, key="mileage")
        
        with col3:
            prod_year = st.number_input("Año", 1980, 2025, 2020, 1, key="year")
            cylinders = st.number_input("Cilindros", 2, 16, 4, 1, key="cyl")
            airbags = st.number_input("Airbags", 0, 16, 4, 1, key="airbag")
            levy = st.number_input("Levy", 0, 10000, 0, 100, key="levy")
            doors = st.selectbox("Puertas", sorted(df_for_prediction['Doors'].dropna().unique()), key="doors")
        
        # Botón de predicción
        if st.button("🔮 Predecir Precio", type="primary", use_container_width=True, key="predict_btn"):
            model = st.session_state.trained_model
            X_template = st.session_state.X_template
            
            # Crear DataFrame con inputs
            if optimize_categories:
                input_data = pd.DataFrame({
                    'Levy': [levy], 'Prod. year': [prod_year], 'Cylinders': [cylinders],
                    'Airbags': [airbags], 'Engine volume': [engine_volume], 'Mileage': [mileage],
                    'Doors': [doors], 'Manufacturer_Grouped': [manufacturer_grouped],
                    'Model_Grouped': [model_grouped], 'Category': [category],
                    'Gear box type': [gear], 'Leather interior': [leather],
                    'Fuel type': [fuel], 'Drive wheels': [drive_wheels], 'Wheel': [wheel]
                })
                columns_to_dummy = ['Manufacturer_Grouped', 'Model_Grouped', 'Gear box type', "Category", 
                                    'Leather interior', 'Fuel type', 'Drive wheels', 'Wheel']
            else:
                input_data = pd.DataFrame({
                    'Levy': [levy], 'Prod. year': [prod_year], 'Cylinders': [cylinders],
                    'Airbags': [airbags], 'Engine volume': [engine_volume], 'Mileage': [mileage],
                    'Doors': [doors], 'Manufacturer': [manufacturer], 'Model': [model],
                    'Category': [category], 'Gear box type': [gear], 'Leather interior': [leather],
                    'Fuel type': [fuel], 'Drive wheels': [drive_wheels], 'Wheel': [wheel]
                })
                columns_to_dummy = ['Manufacturer', 'Model', 'Gear box type', "Category", 
                                    'Leather interior', 'Fuel type', 'Drive wheels', 'Wheel']
            
            # One-hot encoding
            input_encoded = pd.get_dummies(input_data, columns=columns_to_dummy, drop_first=True)
            
            # Limpiar nombres de columnas
            input_encoded.columns = input_encoded.columns.str.replace('[', '_', regex=False)
            input_encoded.columns = input_encoded.columns.str.replace(']', '_', regex=False)
            input_encoded.columns = input_encoded.columns.str.replace('<', '_', regex=False)
            input_encoded.columns = input_encoded.columns.str.replace('>', '_', regex=False)
            input_encoded.columns = input_encoded.columns.str.replace(',', '_', regex=False)
            
            # Aplicar PCA si estaba activo
            use_pca = config.get('use_pca', False)
            if use_pca and st.session_state.get('scaler') is not None:
                scaler = st.session_state.scaler
                pca = st.session_state.pca
                
                numeric_cols = ['Levy', 'Prod. year', 'Cylinders', 'Airbags', 
                               'Engine volume', 'Mileage', 'Doors']
                
                input_numeric = input_encoded[numeric_cols]
                input_categorical_cols = [col for col in input_encoded.columns if col not in numeric_cols]
                input_categorical = input_encoded[input_categorical_cols]
                
                input_scaled = scaler.transform(input_numeric)
                input_pca = pca.transform(input_scaled)
                
                n_components = config.get('n_components', 6)
                pca_columns = [f'PC{i+1}' for i in range(n_components)]
                input_pca_df = pd.DataFrame(input_pca, columns=pca_columns, index=input_encoded.index)
                
                input_encoded = pd.concat([input_pca_df, input_categorical], axis=1)
            
            # Alinear columnas
            for col in X_template.columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[X_template.columns]
            
            # Hacer predicción
            prediction = model.predict(input_encoded)[0]
            
            # Mostrar resultados
            st.success("✅ Predicción Completada")
            
            result_cols = st.columns(3)
            with result_cols[0]:
                st.metric("💰 Precio Predicho", f"${prediction:,.0f}")
            with result_cols[1]:
                st.metric("📊 R² Score", f"{metrics['R2']:.4f}")
            with result_cols[2]:
                st.metric("📈 R² Ajustado", f"{metrics['Adjusted R2']:.4f}")
    
    else:
        st.info("👆 Primero configura y entrena un modelo para poder hacer predicciones")

# ==================== TAB 2: ANALÍTICA DEL MODELO ====================
with tab2:
    st.header("Analítica y Diagnóstico del Modelo")
    
    if st.session_state.get('train_analytics', False):
        config = st.session_state.model_config
        metrics = st.session_state.model_metrics
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        
        # Configuración
        st.info(f"""
        **Configuración del Modelo:**
        - Algoritmo: {config['model_type']}
        - PCA: {'Sí' if config['use_pca'] else 'No'} {f"({config['n_components']} componentes)" if config['use_pca'] else ''}
        - Optimización: {'Sí' if config['optimize_categories'] else 'No'}
        - Profundidad: {config.get('max_depth', 'Sin límite')}
        """)
        
        # Métricas
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("R²", f"{metrics['R2']:.4f}")
        with metric_cols[1]:
            st.metric("R² Ajustado", f"{metrics['Adjusted R2']:.4f}")
        with metric_cols[2]:
            st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
        with metric_cols[3]:
            st.metric("MAE", f"${metrics['MAE']:,.0f}")
        with metric_cols[4]:
            st.metric("Features", metrics['Num Features'])
        
        if metrics.get('Variance Explained (PCA)'):
            st.success(f"📊 Varianza PCA: {metrics['Variance Explained (PCA)']*100:.2f}%")
        
        st.divider()
        
        # Gráficos
        comparison_df = pd.DataFrame({
            'Real': y_test.values,
            'Predicción': y_pred,
            'Error': y_test.values - y_pred,
            'Error %': ((y_test.values - y_pred) / y_test.values * 100)
        })
        
        viz_tab1, viz_tab2 = st.tabs(["Scatter Plot", "Residuos"])
        
        with viz_tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test.values, y=y_pred, mode='markers',
                marker=dict(size=6, color=comparison_df['Error %'], colorscale='RdYlGn_r',
                           showscale=True, colorbar=dict(title="Error %")),
                text=[f"Real: ${r:,.0f}<br>Pred: ${p:,.0f}" for r, p in zip(y_test.values, y_pred)]
            ))
            
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', line=dict(color='red', dash='dash')))
            
            fig.update_layout(title="Predicciones vs Reales", xaxis_title="Real ($)",
                             yaxis_title="Predicho ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=comparison_df['Error'], mode='markers',
                                    marker=dict(size=6, color=comparison_df['Error'], colorscale='RdYlGn')))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(title="Residuos", xaxis_title="Predicho ($)",
                             yaxis_title="Error ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        st.subheader("Estadísticas de Errores")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.write(f"**Media Error:** ${comparison_df['Error'].abs().mean():,.2f}")
            st.write(f"**Mediana Error:** ${comparison_df['Error'].abs().median():,.2f}")
        with stat_col2:
            st.write(f"**Media Error %:** {comparison_df['Error %'].abs().mean():.2f}%")
            st.write(f"**Mediana Error %:** {comparison_df['Error %'].abs().median():.2f}%")
    
    else:
        st.info("👈 Primero entrena un modelo en la pestaña 'Predicción'")

# Footer
st.divider()
st.caption("🚗 Car Price Prediction | Powered by XGBoost & Streamlit")

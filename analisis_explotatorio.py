# -*- coding: utf-8 -*-
"""
Analisis PCA y Optimizacion de Variables Categoricas
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==================== CARGA Y LIMPIEZA DE DATOS ====================
print("=" * 80)
print("ANALISIS DE OPTIMIZACION DE FEATURES")
print("=" * 80)

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

mean_price = df['Price'].mean()
std_price = df['Price'].std()
upper_bound = mean_price + (3 * std_price)
df = df[df['Price'] < upper_bound]

print(f"\n** Datos cargados y limpiados: {df.shape[0]} registros, {df.shape[1]} columnas")

# ==================== ANALISIS DE VARIABLES CATEGORICAS ====================
print("\n" + "=" * 80)
print("ANALISIS DE VARIABLES CATEGORICAS")
print("=" * 80)

categorical_vars = ['Manufacturer', 'Model', 'Gear box type', 'Category', 
                    'Leather interior', 'Fuel type', 'Drive wheels', 'Wheel', 'Color']

for var in categorical_vars:
    unique_count = df[var].nunique()
    print(f"\n{var}: {unique_count} categorias unicas")
    
    if unique_count > 20:
        print(f"  !! ALTA CARDINALIDAD - Requiere agrupacion")
        value_counts = df[var].value_counts()
        print(f"  Top 10 mas frecuentes:")
        for i, (val, count) in enumerate(value_counts.head(10).items(), 1):
            print(f"    {i}. {val}: {count} ({count/len(df)*100:.2f}%)")
    else:
        value_counts = df[var].value_counts()
        print(f"  Distribucion:")
        for val, count in value_counts.items():
            print(f"    - {val}: {count} ({count/len(df)*100:.2f}%)")

# ==================== ANALISIS DE MANUFACTURER ====================
print("\n" + "=" * 80)
print("ANALISIS DETALLADO: MANUFACTURER")
print("=" * 80)

manufacturer_stats = df.groupby('Manufacturer').agg({
    'Price': ['mean', 'count'],
    'ID': 'count'
}).round(2)

manufacturer_stats.columns = ['Precio_Medio', 'Cantidad', 'Total']
manufacturer_stats = manufacturer_stats.sort_values('Precio_Medio', ascending=False)

print("\nTodas las marcas ordenadas por precio medio:")
print(manufacturer_stats)

# Definir criterios de agrupacion
threshold_count = 50  # Menos de 50 vehiculos se considera para agrupacion
price_threshold_high = 30000  # Precio promedio alto
price_threshold_low = 15000   # Precio promedio bajo

manufacturer_stats['Categoria'] = 'Mantener'
manufacturer_stats.loc[
    (manufacturer_stats['Cantidad'] < threshold_count) & 
    (manufacturer_stats['Precio_Medio'] >= price_threshold_high), 
    'Categoria'
] = 'Otros - Gama Alta'

manufacturer_stats.loc[
    (manufacturer_stats['Cantidad'] < threshold_count) & 
    (manufacturer_stats['Precio_Medio'] < price_threshold_high) &
    (manufacturer_stats['Precio_Medio'] >= price_threshold_low), 
    'Categoria'
] = 'Otros - Gama Media'

manufacturer_stats.loc[
    (manufacturer_stats['Cantidad'] < threshold_count) & 
    (manufacturer_stats['Precio_Medio'] < price_threshold_low), 
    'Categoria'
] = 'Otros - Gama Baja'

print("\n" + "-" * 80)
print("PROPUESTA DE AGRUPACION DE MANUFACTURER:")
print("-" * 80)

for categoria in ['Mantener', 'Otros - Gama Alta', 'Otros - Gama Media', 'Otros - Gama Baja']:
    marcas = manufacturer_stats[manufacturer_stats['Categoria'] == categoria]
    if len(marcas) > 0:
        print(f"\n{categoria} ({len(marcas)} marcas):")
        for marca in marcas.index:
            precio = marcas.loc[marca, 'Precio_Medio']
            cantidad = marcas.loc[marca, 'Cantidad']
            print(f"  - {marca}: ${precio:,.0f} ({int(cantidad)} vehiculos)")

# ==================== ANALISIS DE MODEL ====================
print("\n" + "=" * 80)
print("ANALISIS DETALLADO: MODEL")
print("=" * 80)

model_stats = df.groupby('Model').agg({
    'Price': ['mean', 'count']
}).round(2)

model_stats.columns = ['Precio_Medio', 'Cantidad']
model_stats = model_stats.sort_values('Cantidad', ascending=False)

print(f"\nTotal de modelos unicos: {len(model_stats)}")
print(f"\nTop 20 modelos mas frecuentes:")
print(model_stats.head(20))

# Modelos con pocas ocurrencias
low_frequency_models = model_stats[model_stats['Cantidad'] < 20]
print(f"\n!! Modelos con menos de 20 ocurrencias: {len(low_frequency_models)} ({len(low_frequency_models)/len(model_stats)*100:.1f}%)")
print(f"   Estos generarian {len(low_frequency_models)} variables dummy con poca informacion")

# Propuesta de agrupacion de modelos
model_stats['Categoria'] = 'Mantener'
model_stats.loc[
    (model_stats['Cantidad'] < 20) & 
    (model_stats['Precio_Medio'] >= 30000), 
    'Categoria'
] = 'Otros Modelos - Gama Alta'

model_stats.loc[
    (model_stats['Cantidad'] < 20) & 
    (model_stats['Precio_Medio'] < 30000) &
    (model_stats['Precio_Medio'] >= 15000), 
    'Categoria'
] = 'Otros Modelos - Gama Media'

model_stats.loc[
    (model_stats['Cantidad'] < 20) & 
    (model_stats['Precio_Medio'] < 15000), 
    'Categoria'
] = 'Otros Modelos - Gama Baja'

for categoria in ['Mantener', 'Otros Modelos - Gama Alta', 'Otros Modelos - Gama Media', 'Otros Modelos - Gama Baja']:
    modelos = model_stats[model_stats['Categoria'] == categoria]
    print(f"\n{categoria}: {len(modelos)} modelos")

# ==================== ANALISIS PCA ====================
print("\n" + "=" * 80)
print("ANALISIS PCA - REDUCCION DE DIMENSIONALIDAD")
print("=" * 80)

# Preparar datos para PCA (solo variables numericas)
numeric_features = ['Levy', 'Prod. year', 'Cylinders', 'Airbags', 
                    'Engine volume', 'Mileage', 'Doors']

df_numeric = df[numeric_features].dropna()
print(f"\nVariables numericas para PCA: {len(numeric_features)}")
print(f"Registros validos: {len(df_numeric)}")

# Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Aplicar PCA con todos los componentes
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Varianza explicada
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nVarianza explicada por componente:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"  PC{i}: {var*100:.2f}% (acumulado: {cum_var*100:.2f}%)")

# Determinar numero óptimo de componentes
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\n>> Componentes necesarios para 90% varianza: {n_components_90}")
print(f">> Componentes necesarios para 95% varianza: {n_components_95}")

# Analisis con 6 componentes (lo solicitado)
pca_6 = PCA(n_components=6)
X_pca_6 = pca_6.fit_transform(X_scaled)
variance_6_components = sum(pca_6.explained_variance_ratio_)

print(f"\n** Con 6 componentes principales se captura: {variance_6_components*100:.2f}% de la varianza")

# Analisis de carga de componentes
print("\n" + "=" * 80)
print("CARGA DE FEATURES EN LOS 6 PRIMEROS COMPONENTES PRINCIPALES")
print("=" * 80)

components_df = pd.DataFrame(
    pca_6.components_,
    columns=numeric_features,
    index=[f'PC{i+1}' for i in range(6)]
)

print("\nMatriz de Componentes (valores absolutos mayores):")
print(components_df.round(3))

# Features mas importantes por componente
print("\n" + "-" * 80)
print("FEATURES MAS INFLUYENTES POR COMPONENTE:")
print("-" * 80)

for i in range(6):
    pc_name = f'PC{i+1}'
    loadings = components_df.loc[pc_name].abs().sort_values(ascending=False)
    print(f"\n{pc_name} (explica {pca_6.explained_variance_ratio_[i]*100:.2f}% varianza):")
    for feature, loading in loadings.head(3).items():
        direction = "+" if components_df.loc[pc_name, feature] > 0 else "-"
        print(f"  {direction} {feature}: {loading:.3f}")

# ==================== CORRELACIONES ====================
print("\n" + "=" * 80)
print("ANALISIS DE CORRELACIONES CON PRECIO")
print("=" * 80)

df_with_price = df[numeric_features + ['Price']].dropna()
correlations = df_with_price.corr()['Price'].drop('Price').sort_values(ascending=False)

print("\nCorrelacion de features numericas con Precio:")
for feature, corr in correlations.items():
    print(f"  {feature}: {corr:.4f}")

# ==================== RECOMENDACIONES FINALES ====================
print("\n" + "=" * 80)
print("RECOMENDACIONES FINALES")
print("=" * 80)

print("\n1. VARIABLES CATEGORICAS:")
print(f"   * Manufacturer: Reducir de {df['Manufacturer'].nunique()} a ~{len(manufacturer_stats[manufacturer_stats['Categoria'] == 'Mantener']) + 3} categorias")
print(f"   * Model: Reducir de {df['Model'].nunique()} a ~{len(model_stats[model_stats['Categoria'] == 'Mantener']) + 3} categorias")
print(f"   * Esto reduciria significativamente las variables dummy generadas")

print("\n2. VARIABLES NUMERICAS:")
print(f"   * Top 6 features por correlacion con Precio:")
for i, (feature, corr) in enumerate(correlations.head(6).items(), 1):
    print(f"     {i}. {feature}: {corr:.4f}")

print("\n3. ESTRATEGIA RECOMENDADA:")
print("   A) Agrupar Manufacturer y Model en categorias de gama")
print("   B) Usar las 6 variables numericas mas correlacionadas")
print("   C) Alternativamente, usar PCA con 6 componentes (captura {:.1f}% varianza)".format(variance_6_components*100))
print("   D) Mantener variables categoricas de baja cardinalidad sin cambios")

print("\n" + "=" * 80)

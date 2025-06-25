import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import os
import joblib

print("--- Fase 3: Preprocesamiento de Datos ---")

# --- Paso 1: Carga de Datos Iniciales ---
print("Cargando el dataset 'simulated_paint_formulas_with_engineered_features.csv'...")
try:
    df = pd.read_csv('data/processed/simulated_paint_formulas_with_engineered_features.csv')
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print("El archivo 'simulated_paint_formulas_with_engineered_features.csv' no fue encontrado en 'data/processed'.")
    print("Asegúrate de que este archivo exista o de generar los datos primero.")
    exit()

# --- Paso 2: Separación de Características y Variable Objetivo ---
X = df.drop('IsSuccess', axis=1)
y = df['IsSuccess']
print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Variable Objetivo): {y.shape}")
print(f"Variable objetivo identificada como 'IsSuccess'.")

# Identificación de columnas numéricas y categóricas
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print(f"\nColumnas Numéricas ({len(numerical_cols)}): {numerical_cols}")
print(f"Columnas Categóricas ({len(categorical_cols)}): {categorical_cols}")


# --- Paso 3: Imputación de Datos Faltantes (si los hay) ---
print("\nIniciando imputación de datos faltantes...")
# Asumiendo imputación para columnas numéricas y categóricas
for col in numerical_cols:
    if X[col].isnull().any():
        imputer_num = SimpleImputer(strategy='mean')
        X[col] = imputer_num.fit_transform(X[[col]]).flatten() # <-- .flatten() añadido
for col in categorical_cols:
    if X[col].isnull().any():
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[col] = imputer_cat.fit_transform(X[[col]]).flatten() # <-- .flatten() añadido

print("Imputación completada. Verificando NaNs restantes:")
print(f"Número total de NaNs en X después de imputación: {X.isnull().sum().sum()}\n")


# --- Paso 4: Preprocesamiento de Características (Codificación One-Hot y Escalado) en el dataset COMPLETO ---
# Creamos un ColumnTransformer para aplicar transformaciones diferentes a tipos de columnas diferentes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols), # Escala las columnas numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) # Codifica las columnas categóricas
    ],
    remainder='passthrough' # Mantiene otras columnas que no estén especificadas (si las hubiera)
)

print("Iniciando aplicación del preprocesador al dataset completo X...")
X_processed_full = preprocessor.fit_transform(X)

# Convertir la matriz numpy resultante del ColumnTransformer a DataFrame
# Primero, obtenemos los nombres de las columnas que saldrán del preprocesador
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_feature_names) # Las numéricas mantienen su nombre

X_processed_full = pd.DataFrame(X_processed_full, columns=all_feature_names, index=X.index)
print("Preprocesamiento del dataset completo X completado.")
print(f"Dimensiones de X_processed_full después de preprocesamiento: {X_processed_full.shape}")
print("Primeras filas del DataFrame X_processed_full:")
print(X_processed_full.head())


# --- Paso 5: División en Conjuntos de Entrenamiento y Prueba (después del preprocesamiento completo) ---
# Asegura una división estratificada para mantener la proporción de clases de 'IsSuccess'
print("\nDividiendo el dataset preprocesado en conjuntos de entrenamiento y prueba (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X_processed_full, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")

print(f"\nProporción de Éxito en y_train antes de balanceo:\n{y_train.value_counts(normalize=True)}")
print(f"\nProporción de Éxito en y_test:\n{y_test.value_counts(normalize=True)}")


# --- Paso 6: Manejo del Desbalanceo con SMOTE (solo en el conjunto de entrenamiento) ---
# Aplicamos SMOTE al conjunto de entrenamiento, X_train y y_train, que ya están preprocesados
print("\nAplicando SMOTE para balancear las clases en el conjunto de ENTRENAMIENTO...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Dimensiones de X_train_resampled después de SMOTE: {X_train_resampled.shape}")
print(f"Dimensiones de y_train_resampled después de SMOTE: {y_train_resampled.shape}")
print(f"\nProporción de Éxito en y_train_resampled (después de SMOTE):\n{y_train_resampled.value_counts(normalize=True)}")
print("Balanceo de clases completado para el conjunto de entrenamiento.")

print("\n--- Preprocesamiento de Datos Completo ---")
print("El dataset está ahora listo para el entrenamiento del modelo de Machine Learning.")


# --- Guardando datos preprocesados para uso futuro ---
print("\n--- Guardando datos preprocesados para uso futuro ---")

output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True) # Crea la carpeta si no existe

# Guarda los datos con SMOTE (para el modelo BASE)
X_train_resampled.to_csv(os.path.join(output_dir, 'X_train_resampled.csv'), index=False)
y_train_resampled.to_csv(os.path.join(output_dir, 'y_train_resampled.csv'), index=False)

# Guarda los datos SIN SMOTE (para el pipeline de GridSearchCV)
# Estas son las variables X_train y y_train RESULTANTES del train_test_split,
# que ya están preprocesadas pero no sobremuestreadas.
X_train.to_csv(os.path.join(output_dir, 'X_train_preprocessed.csv'), index=False) # Renombrado para claridad
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)

# Guarda el conjunto de prueba (ya preprocesado)
X_test.to_csv(os.path.join(output_dir, 'X_test_preprocessed.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print(f"Todos los datos de entrenamiento/prueba guardados exitosamente en: {output_dir}\n")

# Guardar el objeto preprocesador para uso futuro (por ejemplo, para nuevas predicciones)
# El preprocesador se ajusta sobre X_processed_full, no sobre X_train_processed
joblib.dump(preprocessor, os.path.join('models', 'preprocessor.joblib'))
print("Preprocesador guardado en 'models/preprocessor.joblib'")

print("\n--- Proceso de Preprocesamiento Completado ---\n")
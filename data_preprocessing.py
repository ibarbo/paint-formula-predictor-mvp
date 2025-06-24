import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE # Asegúrate de haber instalado: pip install imbalanced-learn

# --- 1. Carga del Dataset ---
# Cargamos nuestro dataset que ya incluye las características de ingeniería.
print("--- Fase 3: Preprocesamiento de Datos ---")
print("Cargando el dataset 'simulated_paint_formulas_with_engineered_features.csv'...")
df = pd.read_csv('simulated_paint_formulas_with_engineered_features.csv')
print("Dataset cargado exitosamente.\n")

# --- 2. Separación de Características (X) y Variable Objetivo (y) ---
# Es fundamental separar las variables de entrada del modelo (X) de la variable
# que queremos predecir (y) antes de cualquier preprocesamiento.
TARGET_COLUMN = 'IsSuccess'
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Variable Objetivo): {y.shape}")
print("Variable objetivo identificada como 'IsSuccess'.\n")

# --- 3. Identificación de Tipos de Columnas ---
# Clasificamos las columnas como numéricas o categóricas para aplicar
# los pasos de preprocesamiento adecuados a cada tipo.
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

print(f"Columnas Numéricas ({len(numerical_cols)}): {numerical_cols}")
print(f"Columnas Categóricas ({len(categorical_cols)}): {categorical_cols}\n")

# --- 4. Manejo de Datos Faltantes (Imputación) ---
# Rellenamos los valores nulos (NaNs) identificados en el EDA.
# Para numéricas: Usamos la mediana, que es robusta a outliers.
# Para categóricas: Usamos la moda ('most_frequent').

print("Iniciando imputación de datos faltantes...")
imputer_numerical = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_numerical.fit_transform(X[numerical_cols])

imputer_categorical = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])

print("Imputación completada. Verificando NaNs restantes:")
print(f"Número total de NaNs en X después de imputación: {X.isnull().sum().sum()}\n") # Debería ser 0

# --- 5. Codificación de Variables Categóricas (One-Hot Encoding) ---
# Convertimos las variables categóricas a un formato numérico que los modelos
# de Machine Learning puedan entender. One-Hot Encoding crea nuevas columnas binarias.

print("Iniciando codificación One-Hot de variables categóricas...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False para obtener un array denso

# Ajustar y transformar las columnas categóricas
X_encoded = encoder.fit_transform(X[categorical_cols])

# Crear un DataFrame con las columnas codificadas y sus nombres apropiados
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# Concatenar las columnas numéricas originales (ahora imputadas) con las categóricas codificadas
X_preprocessed = pd.concat([X[numerical_cols], X_encoded_df], axis=1)

print(f"Dimensiones de X después de la codificación: {X_preprocessed.shape}")
print("Codificación One-Hot completada. Primeras filas del DataFrame preprocesado:\n", X_preprocessed.head())
print("\n")

# --- 6. Escalado de Características Numéricas (StandardScaler) ---
# Normalizamos las características numéricas para que tengan una media de 0 y desviación
# estándar de 1. Esto es importante para algoritmos sensibles a la escala de las características.

print("Iniciando escalado de características numéricas (StandardScaler)...")
scaler = StandardScaler()
X_preprocessed[numerical_cols] = scaler.fit_transform(X_preprocessed[numerical_cols])

print("Escalado completado. Resumen estadístico de columnas numéricas escaladas (media ~0, std ~1):\n", X_preprocessed[numerical_cols].describe())
print("Primeras filas del DataFrame preprocesado (valores numéricos escalados):\n", X_preprocessed.head())
print("\n")

# --- 7. División del Dataset (Entrenamiento y Prueba) ---
# Dividimos el dataset en conjuntos de entrenamiento (para que el modelo aprenda)
# y prueba (para evaluar el modelo con datos no vistos). Usamos stratify=y
# para mantener la proporción de clases de 'IsSuccess' en ambos conjuntos.

print("Dividiendo el dataset en conjuntos de entrenamiento y prueba (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")

print("\nProporción de Éxito en y_train antes de balanceo:")
print(y_train.value_counts(normalize=True))
print("\nProporción de Éxito en y_test:")
print(y_test.value_counts(normalize=True))
print("\n")

# --- 8. Manejo del Desbalance de Clases (SMOTE) ---
# Aplicamos SMOTE para sobremuestrear la clase minoritaria ('IsSuccess') en el
# conjunto de ENTRENAMIENTO solamente. Esto ayuda al modelo a aprender mejor
# sobre la clase de interés sin introducir sesgos en la evaluación.

print("Aplicando SMOTE para balancear las clases en el conjunto de ENTRENAMIENTO...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Dimensiones de X_train después de SMOTE: {X_train_resampled.shape}")
print(f"Dimensiones de y_train después de SMOTE: {y_train_resampled.shape}")

print("\nProporción de Éxito en y_train_resampled (después de SMOTE):")
print(y_train_resampled.value_counts(normalize=True))
print("\nBalanceo de clases completado para el conjunto de entrenamiento.\n")

print("--- Preprocesamiento de Datos Completo ---")
print("El dataset está ahora listo para el entrenamiento del modelo de Machine Learning.")

# Puedes guardar los conjuntos preprocesados si lo deseas, aunque para este workflow
# es común pasarlos directamente a la fase de Modelado.
# X_train_resampled.to_csv('X_train_preprocessed.csv', index=False)
# y_train_resampled.to_csv('y_train_resampled.csv', index=False)
# X_test.to_csv('X_test_preprocessed.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)
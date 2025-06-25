# scripts/model_training/train_simple_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import sys

print("--- Iniciando Entrenamiento de Modelo Simple para SHAP Debug ---")

# --- Configuración de rutas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_FILE_NAME = 'simulated_paint_formulas_with_engineered_features.csv'
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', DATA_FILE_NAME)

# Rutas para guardar el modelo y preprocesador SIMPLE
SIMPLE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model')
SIMPLE_MODEL_PATH = os.path.join(SIMPLE_MODEL_DIR, 'xgb_simple_model.joblib')
SIMPLE_PREPROCESSOR_PATH = os.path.join(SIMPLE_MODEL_DIR, 'simple_preprocessor.joblib')

# Crear el directorio si no existe
try:
    os.makedirs(SIMPLE_MODEL_DIR, exist_ok=True)
    print(f"Directorio de modelo simple verificado/creado en: {SIMPLE_MODEL_DIR}")
except OSError as e:
    print(f"Error al crear el directorio {SIMPLE_MODEL_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

# --- Cargar datos ---
try:
    df_raw = pd.read_csv(PROCESSED_DATA_PATH, dtype={
        "ResinPercentage": float, "PigmentPercentage": float, "SolventPercentage": float, "AdditivePercentage": float,
        "ApplicationTemp_C": float, "Humidity": float, "PHLevel": float, "Viscosity": float,
        "DryingTime_Hours": float, "Coverage": float, "Gloss": float, "Biocide_Percentage": float,
        "Coalescent_Percentage": float, "Defoamer_Percentage": float, "Dispersant_Percentage": float,
        "EstimatedDensity": float, "ResinToPigmentRatio": float, "ResinToSolventRatio": float,
        "PigmentToSolventRatio": float, "Surfactant_Percentage": float, "Thickener_Percentage": float,
        "TotalAdditivesPercentage": float,
        # Columnas binarias/indicadoras - TRATARLAS COMO FLOAT SI LO ESPERA TU MODELO
        "AcrylicOnWood": float, "EpoxyOnMetal": float, "HighDryingTime": float, "LowApplicationTemp": float, "TiO2OnConcrete": float,
        
        # COLUMNAS CATEGÓRICAS - ESTO ES CRÍTICO
        "SubstrateType": str, "ApplicationMethod": str, "Biocide_Supplier": str,
        "Coalescent_Supplier": str, "Defoamer_Supplier": str, "Dispersant_Supplier": str,
        "HidingPower": str, # <-- ASEGURAR QUE HidingPower ES STRING AQUÍ
        "PigmentSupplier": str, "PigmentType": str, "ResinSupplier": str,
        "ResinType": str, "SolventSupplier": str, "SolventType": str,
        "Surfactant_Supplier": str, "Thickener_Supplier": str
    })
    print(f"Datos raw cargados exitosamente desde: {PROCESSED_DATA_PATH}")
except Exception as e:
    print(f"ERROR: No se pudieron cargar los datos o los tipos de datos no coinciden: {e}", file=sys.stderr)
    sys.exit(1)

TARGET_COLUMN = 'IsSuccess'
if TARGET_COLUMN in df_raw.columns:
    X = df_raw.drop(TARGET_COLUMN, axis=1)
    y = df_raw[TARGET_COLUMN]
    print(f"Columna objetivo '{TARGET_COLUMN}' separada.")
else:
    print(f"ERROR: Columna objetivo '{TARGET_COLUMN}' no encontrada.", file=sys.stderr)
    sys.exit(1)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Datos divididos: X_train {X_train.shape}, X_test {X_test.shape}")

# --- Definir preprocesador (ColumnTransformer) ---
# REVISIÓN CRÍTICA: DEFINIR EXPLÍCITAMENTE LAS COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# Asegúrate de que estas listas sean exhaustivas y correctas según tus datos.
# Las columnas 'binarias' que son 0 o 1 (ej. AcrylicOnWood) pueden ir en numerical_cols si se tratan como números.
# Si contienen cadenas como 'True'/'False' o cualquier otra categoría, deben ir en categorical_cols.
# Basado en tu script, asumo que tus 'binarias' son numéricas (0.0 o 1.0).

# Columnas numéricas (las que recibirán StandardScaler)
numerical_cols = [
    'ResinPercentage', 'PigmentPercentage', 'SolventPercentage', 'AdditivePercentage', # <--- Ahora existirán
    'ApplicationTemp_C', 'Humidity', 'PHLevel', 'Viscosity', 'DryingTime_Hours', # <--- Ahora existirán
    'Coverage', 'Gloss', 'Biocide_Percentage', 'Coalescent_Percentage', # <--- Ahora existirán
    'Defoamer_Percentage', 'Dispersant_Percentage', 'EstimatedDensity',
    'ResinToPigmentRatio', 'ResinToSolventRatio', 'PigmentToSolventRatio',
    'Surfactant_Percentage', 'Thickener_Percentage', 'TotalAdditivesPercentage',
    # Binarias/Indicadoras (si son numéricas, ej. 0.0 o 1.0)
    'AcrylicOnWood', 'EpoxyOnMetal', 'HighDryingTime', 'LowApplicationTemp', 'TiO2OnConcrete'
]

# Columnas categóricas (las que recibirán OneHotEncoder)
categorical_cols = [
    'SubstrateType', 'ApplicationMethod', # <--- Ahora existirá
    'Biocide_Supplier', 'Coalescent_Supplier',
    'Defoamer_Supplier', 'Dispersant_Supplier', 'HidingPower', # <-- HIDINGPOWER ES STRING AHORA DESDE GENERACION
    'PigmentSupplier', 'PigmentType', 'ResinSupplier', 'ResinType',
    'SolventSupplier', 'SolventType', 'Surfactant_Supplier', 'Thickener_Supplier'
]

# AÑADIDO: Ordenar explícitamente las columnas de X_train y X_test
expected_features_order = numerical_cols + categorical_cols
# Filtrar y reordenar X_train
X_train = X_train[expected_features_order]
X_test = X_test[expected_features_order]


# Verificar que no haya solapamiento y que todas las columnas de X estén cubiertas
all_features_in_df = set(X_train.columns) # Obtener las columnas reales en el DataFrame
all_expected_features = set(numerical_cols + categorical_cols) # Obtener las columnas esperadas por las listas

if all_expected_features != all_features_in_df:
    print("ADVERTENCIA: Las columnas reales del DataFrame no coinciden exactamente con las listas numéricas/categóricas.", file=sys.stderr)
    print(f"Columnas en DataFrame pero no en listas (INESPERADAS): {all_features_in_df - all_expected_features}")
    print(f"Columnas en listas pero no en DataFrame (FALTANTES): {all_expected_features - all_features_in_df}")
    # sys.exit(1) # Podrías querer salir aquí si esto ocurre

print(f"Dimensiones de X_train después de ordenar: {X_train.shape}")


# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols) # sparse_output=False es útil para SHAP y depuración
    ],
    remainder='drop' # 'drop' es más seguro si ya definimos todas las columnas
)
print("ColumnTransformer preprocesador definido.")

# --- Definir el Modelo Simple (XGBoost) ---
simple_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, 
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
print("Modelo XGBoost simple definido.")

# --- Crear Pipeline de Preprocesamiento + Modelo ---
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', simple_model)])
print("Pipeline de preprocesamiento y modelo simple creado.")

# --- Entrenar el Pipeline ---
print("Entrenando el modelo simple (esto será rápido)...")
try:
    full_pipeline.fit(X_train, y_train)
    print("Modelo simple entrenado exitosamente.")
except Exception as e:
    print(f"ERROR: Fallo al entrenar el modelo simple: {e}", file=sys.stderr)
    sys.exit(1)

# --- Guardar el modelo y el preprocesador ---
try:
    joblib.dump(full_pipeline, SIMPLE_MODEL_PATH)
    print(f"Modelo simple guardado en: {SIMPLE_MODEL_PATH}")
    joblib.dump(preprocessor, SIMPLE_PREPROCESSOR_PATH)
    print(f"Preprocesador simple guardado en: {SIMPLE_PREPROCESSOR_PATH}")
except Exception as e:
    print(f"ERROR: Fallo al guardar el modelo/preprocesador simple: {e}", file=sys.stderr)
    sys.exit(1)

print("\n--- Entrenamiento de Modelo Simple Completado ---")
print("Puedes usar este modelo para depurar el análisis SHAP.")
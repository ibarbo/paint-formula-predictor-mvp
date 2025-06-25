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
    df_raw = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Datos raw cargados exitosamente desde: {PROCESSED_DATA_PATH}")
except Exception as e:
    print(f"ERROR: No se pudieron cargar los datos: {e}", file=sys.stderr)
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
# Identificar columnas numéricas y categóricas
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Mantiene las columnas no especificadas si las hay
)
print("ColumnTransformer preprocesador definido.")

# --- Definir el Modelo Simple (XGBoost) ---
# Usamos un XGBoost más pequeño y rápido para este test
simple_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, # Necesario para XGBoost 1.x con eval_metric
    n_estimators=50,        # Menos árboles
    max_depth=3,            # Menor profundidad
    learning_rate=0.1,      # Estándar
    random_state=42,
    n_jobs=-1               # Usar todos los cores disponibles
)
print("Modelo XGBoost simple definido.")

# --- Crear Pipeline de Preprocesamiento + Modelo ---
# No usaremos imbalanced-learn en este pipeline simple para evitar dependencias adicionales
# y centrarnos en el flujo SHAP.
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
    # Guardar el preprocesador por separado también, para flexibilidad, aunque esté en el pipeline.
    joblib.dump(preprocessor, SIMPLE_PREPROCESSOR_PATH)
    print(f"Preprocesador simple guardado en: {SIMPLE_PREPROCESSOR_PATH}")
except Exception as e:
    print(f"ERROR: Fallo al guardar el modelo/preprocesador simple: {e}", file=sys.stderr)
    sys.exit(1)

print("\n--- Entrenamiento de Modelo Simple Completado ---")
print("Puedes usar este modelo para depurar el análisis SHAP.")
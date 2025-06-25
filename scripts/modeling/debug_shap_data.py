import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt # Añadir esta importación si no está
import os
import sys
from sklearn.model_selection import train_test_split # Asegurarse de que esté


print("--- DEBUGGING SHAP DATA FLOW (Usando Modelo Simple) ---")

# --- Configuración de rutas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# --- RUTAS AL MODELO Y PREPROCESADOR SIMPLES ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model', 'xgb_simple_model.joblib')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model', 'simple_preprocessor.joblib')

DATA_FILE_NAME = 'simulated_paint_formulas_with_engineered_features.csv'
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', DATA_FILE_NAME)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'shap_simple_debug') # Nuevo directorio para figuras de depuración

try:
    os.makedirs(FIGURES_DIR, exist_ok=True)
except OSError as e:
    print(f"Error al crear el directorio de figuras {FIGURES_DIR}: {e}", file=sys.stderr)
    sys.exit(1)


print(f"DEBUG: PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DEBUG: MODEL_PATH (Simple): {MODEL_PATH}")
print(f"DEBUG: PREPROCESSOR_PATH (Simple): {PREPROCESSOR_PATH}")
print(f"DEBUG: PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")

# --- 1. Cargar el modelo (simple) ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"DEBUG: Modelo simple cargado exitosamente. Tipo: {type(model)}")
    if hasattr(model, 'named_steps') and "classifier" in model.named_steps:
        print(f"DEBUG: Modelo es un Pipeline. Clasificador: {type(model.named_steps['classifier'])}")
    else:
        print(f"DEBUG: Modelo es el clasificador directo: {type(model)}")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo simple desde {MODEL_PATH}. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. Cargar el preprocesador (simple) ---
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"DEBUG: Preprocesador simple cargado exitosamente. Tipo: {type(preprocessor)}")
except Exception as e:
    print(f"ERROR: No se pudo cargar el preprocesador simple desde {PREPROCESSOR_PATH}. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. Cargar los datos raw ---
try:
    df_raw = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"DEBUG: Datos raw cargados exitosamente. Dimensiones: {df_raw.shape}")
    # ... (resto de prints de head y dtypes si quieres, son útiles)
except Exception as e:
    print(f"ERROR: No se pudieron cargar los datos raw desde {PROCESSED_DATA_PATH}. Error: {e}", file=sys.stderr)
    sys.exit(1)

TARGET_COLUMN = 'IsSuccess'
if TARGET_COLUMN in df_raw.columns:
    X_raw = df_raw.drop(TARGET_COLUMN, axis=1)
    y = df_raw[TARGET_COLUMN]
    print(f"DEBUG: Columna objetivo '{TARGET_COLUMN}' separada.")
else:
    print(f"ERROR: Columna objetivo '{TARGET_COLUMN}' no encontrada.", file=sys.stderr)
    sys.exit(1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42, stratify=y)
print(f"DEBUG: X_test_raw dimensiones: {X_test_raw.shape}")
# ... (resto de prints de head y dtypes si quieres)

# --- 4. Aplicar el preprocesador ---
print("DEBUG: Aplicando preprocesador a X_test_raw...")
try:
    X_test_preprocessed_array = preprocessor.transform(X_test_raw)
    print(f"DEBUG: X_test_preprocessed_array dimensiones: {X_test_preprocessed_array.shape}")
    print(f"DEBUG: Tipo de datos de X_test_preprocessed_array: {X_test_preprocessed_array.dtype}")
except Exception as e:
    print(f"ERROR: Fallo al aplicar el preprocesador: {e}. Asegúrate de que las columnas en tus datos raw coinciden con las usadas para entrenar el preprocesador.", file=sys.stderr)
    sys.exit(1)

# --- 5. Crear DataFrame final para SHAP con nombres de columnas ---
print("DEBUG: Creando DataFrame X_test para SHAP...")
feature_names = []
try:
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        print("DEBUG: Nombres de características obtenidos de get_feature_names_out().")
    else:
        feature_names = [f'feature_{i}' for i in range(X_test_preprocessed_array.shape[1])]
        print("DEBUG: Nombres de características obtenidos manualmente (genéricos).")

    X_test = pd.DataFrame(X_test_preprocessed_array, columns=feature_names, index=X_test_raw.index)
    print(f"DEBUG: X_test (para SHAP) dimensiones: {X_test.shape}")
    print("DEBUG: Primeras 5 filas de X_test (para SHAP):")
    print(X_test.head())
    print("DEBUG: Tipos de datos de X_test (para SHAP):")
    print(X_test.dtypes.value_counts())

except Exception as e:
    print(f"ERROR: Fallo al crear DataFrame X_test para SHAP o obtener nombres de características. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- 6. Inicializar el SHAP Explainer ---
print("DEBUG: Inicializando SHAP Explainer...")
try:
    # Usar model.named_steps["classifier"] para acceder al XGBoost dentro del Pipeline
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    print("DEBUG: SHAP Explainer inicializado con clasificador desde Pipeline.")
except Exception as e:
    print(f"ERROR: Fallo al inicializar SHAP Explainer. Error: {e}", file=sys.stderr)
    sys.exit(1)


# --- 7. Calcular SHAP values con muestra muy pequeña ---
print("DEBUG: Intentando calcular SHAP values con una muestra muy pequeña (5-10 filas)...")
N_SHAP_TEST_SAMPLES = 10 # MUY PEQUEÑO para depurar
if X_test.shape[0] > N_SHAP_TEST_SAMPLES:
    X_test_for_shap_debug = X_test.sample(n=N_SHAP_TEST_SAMPLES, random_state=42).copy()
    print(f"DEBUG: Usando una muestra de {X_test_for_shap_debug.shape[0]} filas para la prueba SHAP.")
else:
    X_test_for_shap_debug = X_test.copy()
    print(f"DEBUG: Usando todas las {X_test_for_shap_debug.shape[0]} filas para la prueba SHAP.")

try:
    # shap_values[1] es para la clase positiva
    shap_values_debug = explainer.shap_values(X_test_for_shap_debug)
    print("DEBUG: ¡Valores SHAP calculados exitosamente para la muestra de depuración!")
    if isinstance(shap_values_debug, list):
        print(f"DEBUG: shap_values_debug[0] shape: {shap_values_debug[0].shape}")
        print(f"DEBUG: shap_values_debug[1] shape: {shap_values_debug[1].shape}")
    else:
        print(f"DEBUG: shap_values_debug shape: {shap_values_debug.shape}")
except Exception as e:
    print(f"ERROR: ¡Fallo al calcular valores SHAP incluso con muestra pequeña! Error: {e}", file=sys.stderr)
    sys.exit(1)


# --- 8. Generar un plot de prueba (opcional, pero buena verificación) ---
print("DEBUG: Generando plot de prueba...")
try:
    # AÑADE ESTAS LÍNEAS DE DEPURACIÓN AQUÍ
    print(f"DEBUG: Tipo de shap_values_debug: {type(shap_values_debug)}")
    if isinstance(shap_values_debug, list):
        print(f"DEBUG: Longitud de la lista shap_values_debug: {len(shap_values_debug)}")
        if len(shap_values_debug) > 0:
            print(f"DEBUG: Tipo de shap_values_debug[0]: {type(shap_values_debug[0])}, Forma: {shap_values_debug[0].shape}")
        if len(shap_values_debug) > 1:
            print(f"DEBUG: Tipo de shap_values_debug[1]: {type(shap_values_debug[1])}, Forma: {shap_values_debug[1].shape}")
    else:
        print(f"DEBUG: Forma de shap_values_debug (si no es lista): {shap_values_debug.shape}")

    print(f"DEBUG: Tipo de X_test_for_shap_debug: {type(X_test_for_shap_debug)}, Forma: {X_test_for_shap_debug.shape}")


    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_debug, X_test_for_shap_debug, show=False)
    plt.title("Test SHAP Plot (Simple Model)")
    test_plot_path = os.path.join(FIGURES_DIR, 'test_shap_plot_simple_model.png')
    plt.savefig(test_plot_path)
    plt.close()
    print(f"DEBUG: Plot de prueba guardado en: {test_plot_path}")
except Exception as e:
    print(f"ERROR: Fallo al generar/guardar el plot de prueba: {e}", file=sys.stderr)

print("\n--- DEBUGGING COMPLETED (Usando Modelo Simple). Check the output for any ERROR messages. ---")
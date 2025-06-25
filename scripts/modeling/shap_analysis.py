import pandas as pd
import numpy as np
import joblib # Para cargar el modelo y el preprocesador guardados
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sys # Para manejar la salida en caso de errores críticos

# --- Configuración de rutas y archivos ---
# Obtener la ruta absoluta del directorio del script actual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Subir dos niveles para llegar a la raíz del proyecto (paint_predictor_mvp)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Rutas de archivos basadas en la estructura que me proporcionaste
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost', 'xgb_model_optimized.joblib')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'preprocessor.joblib') # Corregido para que esté dentro de 'xgboost'
DATA_FILE_NAME = 'simulated_paint_formulas_with_engineered_features.csv'
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', DATA_FILE_NAME)
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'shap_individual_plots')

# Crear el directorio para guardar las figuras si no existe
try:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Directorio de figuras verificado/creado en: {FIGURES_DIR}")
except OSError as e:
    print(f"Error al crear el directorio de figuras {FIGURES_DIR}: {e}", file=sys.stderr)
    sys.exit(1) # Salir si no se puede crear el directorio

print("--- Iniciando Análisis SHAP ---")

# --- 1. Cargar el modelo XGBoost optimizado ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Modelo no encontrado en {MODEL_PATH}. Verifica la ruta y el nombre del archivo.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado al cargar el modelo: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. Cargar el preprocesador ---
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Preprocesador cargado exitosamente desde: {PREPROCESSOR_PATH}")
except FileNotFoundError:
    print(f"Error: Preprocesador no encontrado en {PREPROCESSOR_PATH}. Verifica la ruta y el nombre del archivo.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado al cargar el preprocesador: {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. Cargar los datos raw (antes del preprocesamiento) ---
try:
    df_raw = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Datos raw cargados exitosamente desde: {PROCESSED_DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Datos raw no encontrados en {PROCESSED_DATA_PATH}. Verifica la ruta y el nombre del archivo.", file=sys.stderr)
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: El archivo de datos {PROCESSED_DATA_PATH} está vacío.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado al cargar los datos: {e}", file=sys.stderr)
    sys.exit(1)

# Separar características (X) y la variable objetivo (y)
TARGET_COLUMN = 'IsSuccess'
if TARGET_COLUMN in df_raw.columns:
    X_raw = df_raw.drop(TARGET_COLUMN, axis=1)
    y = df_raw[TARGET_COLUMN]
    print(f"Columna objetivo '{TARGET_COLUMN}' encontrada y separada.")
else:
    print(f"Error: Columna objetivo '{TARGET_COLUMN}' no encontrada en los datos. Por favor, verifica el nombre de la columna.", file=sys.stderr)
    sys.exit(1)

# Dividir los datos ANTES de preprocesar X_raw
# Esto asegura que X_test_raw tiene las columnas originales, y X_test tendrá las preprocesadas.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42, stratify=y)
print(f"Datos divididos en entrenamiento ({X_train_raw.shape[0]} muestras) y prueba ({X_test_raw.shape[0]} muestras).")
print(f"Dimensiones de X_test_raw (antes del preprocesamiento): {X_test_raw.shape}")

# --- 4. Aplicar el preprocesador a X_test_raw ---
print("Aplicando preprocesador a X_test_raw...")
try:
    X_test_preprocessed_array = preprocessor.transform(X_test_raw)
except Exception as e:
    print(f"Error al aplicar el preprocesador: {e}. Asegúrate de que las columnas en tus datos raw coinciden con las usadas para entrenar el preprocesador.", file=sys.stderr)
    sys.exit(1)

# Intentar obtener los nombres de las características post-preprocesamiento para un DataFrame
# Esto es CRÍTICO para que SHAP muestre nombres significativos.
feature_names = []
try:
    # Intenta obtener los nombres de las características si el preprocesador es un ColumnTransformer
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback si no tiene get_feature_names_out, asume nombres originales o numéricos
        print("Advertencia: El preprocesador no tiene 'get_feature_names_out'. Los nombres de las características en SHAP podrían ser genéricos (ej. 'x0', 'x1').", file=sys.stderr)
        # Si tu preprocesador es un Pipeline con OneHotEncoder, puede ser más complejo obtenerlos.
        # Puedes intentar obtenerlos del OneHotEncoder directamente si lo tienes en un paso nombrado.
        # Ejemplo: feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
        feature_names = [f'feature_{i}' for i in range(X_test_preprocessed_array.shape[1])]

    X_test = pd.DataFrame(X_test_preprocessed_array, columns=feature_names, index=X_test_raw.index)
except Exception as e:
    print(f"Error al obtener nombres de características o crear DataFrame post-preprocesamiento: {e}. SHAP podría usar nombres genéricos.", file=sys.stderr)
    X_test = pd.DataFrame(X_test_preprocessed_array, index=X_test_raw.index) # Crear DataFrame sin nombres de columna específicos
finally:
    print(f"Dimensiones de X_test (después del preprocesamiento): {X_test.shape}")


# --- 5. Inicializar el SHAP Explainer ---
# Para modelos basados en árboles (como XGBoost), shap.TreeExplainer es el más eficiente.
# Se adapta si el modelo es un Pipeline o el clasificador directo.
if hasattr(model, 'named_steps') and "classifier" in model.named_steps:
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    print("SHAP TreeExplainer inicializado con clasificador desde Pipeline.")
else:
    explainer = shap.TreeExplainer(model)
    print("SHAP TreeExplainer inicializado directamente con el modelo.")


# --- 6. Calcular los valores SHAP ---
print("Calculando valores SHAP (esto puede tardar unos minutos o más, dependiendo de los recursos)...")

# Reducir la muestra para el cálculo SHAP para manejar problemas de memoria
N_SHAP_SAMPLES = 100 # Intenta con 500. Si aún hay problemas, bájalo a 200 o 100.
if X_test.shape[0] > N_SHAP_SAMPLES:
    X_test_for_shap = X_test.sample(n=N_SHAP_SAMPLES, random_state=42).copy()
    print(f"Usando una muestra de {X_test_for_shap.shape[0]} filas para el cálculo SHAP para ahorrar memoria.")
else:
    X_test_for_shap = X_test.copy()
    print(f"Usando todas las {X_test_for_shap.shape[0]} filas para el cálculo SHAP.")

try:
    # shap_values[1] es para la clase positiva ('Éxito' si es 1) en modelos binarios
    shap_values = explainer.shap_values(X_test_for_shap)
    print("Valores SHAP calculados exitosamente.")
except Exception as e:
    print(f"Error al calcular los valores SHAP: {e}. Considera reducir 'N_SHAP_SAMPLES' aún más o verificar tu instalación de SHAP/XGBoost.", file=sys.stderr)
    sys.exit(1)


# --- 7. Visualizaciones SHAP ---

# 7.1. SHAP Summary Plot (Importancia Global de Características)
print("Generando SHAP Summary Plot (Global Feature Importance)...")
try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X_test_for_shap, show=False) # Usa la muestra reducida
    plt.title('SHAP Global Feature Importance (Class: Success)')
    plt.tight_layout()
    summary_plot_path = os.path.join(FIGURES_DIR, 'shap_global_feature_importance.png')
    plt.savefig(summary_plot_path)
    plt.close() # Cierra la figura para liberar memoria
    print(f"SHAP Summary Plot guardado en: {summary_plot_path}")
except Exception as e:
    print(f"Error al generar/guardar SHAP Summary Plot: {e}", file=sys.stderr)

# 7.2. SHAP Force Plot para instancias individuales (Ejemplo de Éxito y Falla)
print("Preparando para generar SHAP Force Plots para instancias individuales...")

# Obtener predicciones para el set de prueba completo (X_test_raw)
# Si el modelo es un Pipeline, se le pasan los datos RAW. Si es solo el clasificador, preprocesados.
if hasattr(model, 'named_steps') and "classifier" in model.named_steps:
    y_pred_proba = model.predict_proba(X_test_raw)[:, 1]
else:
    y_pred_proba = model.predict_proba(X_test_for_shap)[:, 1] # Usamos la muestra para consistencia

y_pred_class = (y_pred_proba > 0.5).astype(int) # Usando umbral 0.5

# Encontrar un ejemplo de Verdadero Positivo (Real Éxito, Predicho Éxito)
# Buscamos en el conjunto de prueba original y luego mapeamos al preprocesado
true_positives_idx_raw = y_test[(y_test == 1) & (y_pred_class == 1)].index.intersection(X_test_for_shap.index)
if not true_positives_idx_raw.empty:
    sample_success_idx = true_positives_idx_raw[0]
    sample_success_data_preprocessed = X_test_for_shap.loc[sample_success_idx] # Asegurarse de que esté en la muestra SHAP
    
    print(f"\nGenerando SHAP Force Plot para una instancia de 'Éxito' (índice original: {sample_success_idx})...")
    try:
        # El force_plot necesita el valor esperado (explainer.expected_value[1])
        # y los valores SHAP para la instancia específica
        # y los datos de la instancia para mostrar los nombres de las características
        shap_values_for_instance = explainer.shap_values(sample_success_data_preprocessed.to_frame().T)[1][0] # Transponer para que sea 1 fila
        shap.force_plot(explainer.expected_value[1], shap_values_for_instance, sample_success_data_preprocessed, matplotlib=True, show=False)
        plt.tight_layout()
        force_plot_success_path = os.path.join(FIGURES_DIR, f'shap_force_plot_success_instance_{sample_success_idx}.png')
        plt.savefig(force_plot_success_path)
        plt.close()
        print(f"SHAP Force Plot para 'Éxito' guardado en: {force_plot_success_path}")
    except Exception as e:
        print(f"Error al generar/guardar SHAP Force Plot para instancia de éxito: {e}", file=sys.stderr)
else:
    print("\nNo se encontró ninguna instancia de Verdadero Positivo en la muestra para generar Force Plot.")

# Encontrar un ejemplo de Verdadero Negativo (Real Falla, Predicho Falla)
true_negatives_idx_raw = y_test[(y_test == 0) & (y_pred_class == 0)].index.intersection(X_test_for_shap.index)
if not true_negatives_idx_raw.empty:
    sample_failure_idx = true_negatives_idx_raw[0]
    sample_failure_data_preprocessed = X_test_for_shap.loc[sample_failure_idx] # Asegurarse de que esté en la muestra SHAP

    print(f"\nGenerando SHAP Force Plot para una instancia de 'Falla' (índice original: {sample_failure_idx})...")
    try:
        shap_values_for_instance = explainer.shap_values(sample_failure_data_preprocessed.to_frame().T)[1][0]
        shap.force_plot(explainer.expected_value[1], shap_values_for_instance, sample_failure_data_preprocessed, matplotlib=True, show=False)
        plt.tight_layout()
        force_plot_failure_path = os.path.join(FIGURES_DIR, f'shap_force_plot_failure_instance_{sample_failure_idx}.png')
        plt.savefig(force_plot_failure_path)
        plt.close()
        print(f"SHAP Force Plot para 'Falla' guardado en: {force_plot_failure_path}")
    except Exception as e:
        print(f"Error al generar/guardar SHAP Force Plot para instancia de falla: {e}", file=sys.stderr)
else:
    print("\nNo se encontró ninguna instancia de Verdadero Negativo en la muestra para generar Force Plot.")

print("\n--- Análisis SHAP Completado ---")
print(f"Revisa las visualizaciones SHAP en: {FIGURES_DIR}")
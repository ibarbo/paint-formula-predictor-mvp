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

# RUTAS MODIFICADAS PARA CARGAR EL MODELO SIMPLE Y SU PREPROCESADOR
# Asegúrate de que estas rutas coincidan con dónde train_simple_model.py guarda los archivos
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model', 'xgb_simple_model.joblib')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model', 'simple_preprocessor.joblib')

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

# --- 1. Cargar el modelo XGBoost optimizado (ahora el simple) ---
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

# --- 4. Aplicar el preprocesador a X_test_raw para SHAP ---
print("Aplicando preprocesador a X_test_raw para SHAP...")
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
N_SHAP_SAMPLES = 500 # Intenta con 500. Si aún hay problemas, bájalo a 200 o 100.
if X_test.shape[0] > N_SHAP_SAMPLES:
    X_test_for_shap = X_test.sample(n=N_SHAP_SAMPLES, random_state=42).copy()
    print(f"Usando una muestra de {X_test_for_shap.shape[0]} filas para el cálculo SHAP para ahorrar memoria.")
else:
    X_test_for_shap = X_test.copy()
    print(f"Usando todas las {X_test_for_shap.shape[0]} filas para el cálculo SHAP.")

try:
    # Para TreeExplainer con modelos binarios, esto puede devolver una lista de 2 arrays (shap_values_class0, shap_values_class1)
    # O, si está configurado para una salida, directamente el array de la clase positiva.
    shap_values = explainer.shap_values(X_test_for_shap)
    print("Valores SHAP calculados exitosamente.")
    
    # --- DEBUG PRINTS para shap_values del Summary Plot ---
    print(f"DEBUG: Tipo de shap_values (para summary plot): {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"DEBUG: shap_values (para summary plot) es una lista. Longitud: {len(shap_values)}")
        for i, val_array in enumerate(shap_values):
            print(f"DEBUG:   shap_values[{i}] Tipo: {type(val_array)}, Forma: {val_array.shape}, Dim: {val_array.ndim}")
    else:
        print(f"DEBUG: shap_values (para summary plot) NO es una lista. Tipo: {type(shap_values)}, Forma: {shap_values.shape}, Dim: {shap_values.ndim}")
    # --- FIN DEBUG PRINTS ---

except Exception as e:
    print(f"Error al calcular los valores SHAP: {e}. Considera reducir 'N_SHAP_SAMPLES' aún más o verificar tu instalación de SHAP/XGBoost.", file=sys.stderr)
    sys.exit(1)


# --- 7. Visualizaciones SHAP ---

# 7.1. SHAP Summary Plot (Importancia Global de Características)
print("Generando SHAP Summary Plot (Global Feature Importance)...")
try:
    plt.figure(figsize=(10, 8))
    
    # Adaptar para el formato de shap_values
    # Según los errores anteriores, es probable que 'shap_values' sea directamente el array 2D
    # para la clase positiva, en lugar de una lista [clase0, clase1].
    if isinstance(shap_values, list):
        # Si aún es una lista, toma el elemento 1 (clase positiva)
        shap_values_to_plot = shap_values[1]
    else:
        # Si no es una lista, ya es el array que necesitamos
        shap_values_to_plot = shap_values

    # Asegurarse de que shap_values_to_plot sea 2D para summary_plot
    if shap_values_to_plot.ndim == 1: # Si es 1D (no debería ser para summary plot con multiples features)
        print(f"Advertencia: shap_values_to_plot es 1D ({shap_values_to_plot.shape}) para Summary Plot. El plot podría no ser el esperado.")
        # Intentamos reformar a (N_samples, 1) para evitar el error 'not a vector',
        # pero esto no mostrará la contribución por feature si el problema es más profundo.
        shap_values_to_plot = shap_values_to_plot.reshape(-1, 1) if shap_values_to_plot.size > 0 else np.array([[]])


    if shap_values_to_plot.size > 0 and X_test_for_shap.shape[1] > 0:
        shap.summary_plot(shap_values_to_plot, X_test_for_shap, show=False)
        plt.title('SHAP Global Feature Importance (Class: Success)')
        plt.tight_layout()
        summary_plot_path = os.path.join(FIGURES_DIR, 'shap_global_feature_importance.png')
        plt.savefig(summary_plot_path)
        plt.close() # Cierra la figura para liberar memoria
        print(f"SHAP Summary Plot guardado en: {summary_plot_path}")
    else:
        print("No hay suficientes datos o características para generar SHAP Summary Plot.")

except Exception as e:
    print(f"Error al generar/guardar SHAP Summary Plot: {e}", file=sys.stderr)


# 7.2. SHAP Force Plot para instancias individuales (Ejemplo de Éxito y Falla)
print("\nPreparando para generar SHAP Force Plots para instancias individuales...")

# Paso crucial: Preprocesar los datos ANTES de pasarlos al clasificador final
# Si el modelo es un Pipeline, extraemos el clasificador final.
# Si es el clasificador directo (no un Pipeline), lo usamos directamente.
if hasattr(model, 'named_steps') and "classifier" in model.named_steps:
    final_classifier = model.named_steps["classifier"]
    print("Extraído el clasificador final del Pipeline para predicción.")
else:
    final_classifier = model
    print("Usando el modelo directamente para predicción (no es un Pipeline).")

# Aplicar el preprocesador al conjunto de prueba completo X_test_raw para la predicción
try:
    X_test_for_prediction = preprocessor.transform(X_test_raw)
    
    # Opcional: Reconstruir DataFrame con nombres de características si el clasificador los espera
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names_out_full = preprocessor.get_feature_names_out()
        X_test_for_prediction = pd.DataFrame(X_test_for_prediction, columns=feature_names_out_full, index=X_test_raw.index)
    elif isinstance(X_test_for_prediction, np.ndarray): # Asegurarse de que sea DataFrame si no se obtuvieron nombres
         X_test_for_prediction = pd.DataFrame(X_test_for_prediction, index=X_test_raw.index)
    
except Exception as e:
    print(f"Error al preprocesar X_test_raw para la predicción: {e}", file=sys.stderr)
    sys.exit(1)

# Ahora, usar el clasificador final directamente para predict_proba
try:
    y_pred_proba = final_classifier.predict_proba(X_test_for_prediction)[:, 1]
    print("Probabilidades de predicción calculadas exitosamente usando el clasificador final.")
except Exception as e:
    print(f"Error al calcular predict_proba con el clasificador final: {e}", file=sys.stderr)
    sys.exit(1)

y_pred_class = (y_pred_proba > 0.5).astype(int) # Usando umbral 0.5

# --- SECCIÓN DE FORCE PLOTS REVISADA PARA MAYOR ROBUSTEZ EN LA SELECCIÓN Y EXTRACCIÓN DE VALORES SHAP ---
# Encontrar un ejemplo de Real Éxito (y_test == 1) dentro de X_test_for_shap
success_samples_in_shap_indices = X_test_for_shap.index[y_test.loc[X_test_for_shap.index] == 1]
if not success_samples_in_shap_indices.empty:
    sample_success_idx = success_samples_in_shap_indices[0] # Tomar el primer índice
    sample_success_data_preprocessed = X_test_for_shap.loc[sample_success_idx]
    
    print(f"\nGenerando SHAP Force Plot para una instancia de 'Éxito Real' (índice original: {sample_success_idx})...")
    try:
        # Calcular SHAP values para la instancia individual.
        instance_shap_values_raw = explainer.shap_values(sample_success_data_preprocessed.to_frame().T)

        # --- DEBUG PRINTS para instance_shap_values_raw (Force Plot) ---
        print(f"DEBUG: Tipo de instance_shap_values_raw (para force plot éxito): {type(instance_shap_values_raw)}")
        if isinstance(instance_shap_values_raw, list):
            print(f"DEBUG: instance_shap_values_raw es una lista. Longitud: {len(instance_shap_values_raw)}")
            for i, val_array in enumerate(instance_shap_values_raw):
                print(f"DEBUG:   instance_shap_values_raw[{i}] Tipo: {type(val_array)}, Forma: {val_array.shape}, Dim: {val_array.ndim}")
            # Si es una lista y el [1] estaba fuera de límites, probablemente solo devuelve la clase positiva en [0]
            if len(instance_shap_values_raw) > 0:
                shap_values_for_instance = instance_shap_values_raw[0] # Usar [0] si hay solo un elemento o el principal
            else:
                raise ValueError("La lista de valores SHAP para la instancia está vacía.")
        else:
            # Si no es una lista, es el array directo de SHAP values
            shap_values_for_instance = instance_shap_values_raw
        # --- FIN DEBUG PRINTS ---

        # Asegurarse de que es un array 1D (aplanar si es 1xN o 0D)
        if isinstance(shap_values_for_instance, np.ndarray):
            if shap_values_for_instance.ndim > 1 and shap_values_for_instance.shape[0] == 1:
                shap_values_for_instance = shap_values_for_instance.flatten()
            elif shap_values_for_instance.ndim == 0: # Caso de array escalar, convertir a 1D
                shap_values_for_instance = np.array([shap_values_for_instance])
        
        # Ajustar expected_value: asumiendo que no es una lista o que el valor relevante está directamente.
        # Si explainer.expected_value es un array o escalar, ya no se necesita [1].
        force_plot_expected_value = explainer.expected_value
        if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0:
            if len(explainer.expected_value) > 0: # Si es un array, tomar el primer/único valor
                force_plot_expected_value = explainer.expected_value[0]
            else: # Si el array está vacío, usar un valor predeterminado o levantar error
                force_plot_expected_value = 0.0 # O explainer.expected_value si quieres mantener el tipo
                print("Advertencia: explainer.expected_value es un array vacío. Usando 0.0 para force_plot.")


        shap.force_plot(force_plot_expected_value, shap_values_for_instance, sample_success_data_preprocessed, matplotlib=True, show=False)
        plt.tight_layout()
        force_plot_success_path = os.path.join(FIGURES_DIR, f'shap_force_plot_real_success_instance_{sample_success_idx}.png')
        plt.savefig(force_plot_success_path)
        plt.close()
        print(f"SHAP Force Plot para 'Éxito Real' guardado en: {force_plot_success_path}")
    except Exception as e:
        print(f"Error al generar/guardar SHAP Force Plot para instancia de éxito: {e}", file=sys.stderr)
else:
    print("\nNo se encontró ninguna instancia de Real Éxito en la muestra SHAP para generar Force Plot. Considera aumentar N_SHAP_SAMPLES.")

# Encontrar un ejemplo de Real Falla (y_test == 0) dentro de X_test_for_shap
failure_samples_in_shap_indices = X_test_for_shap.index[y_test.loc[X_test_for_shap.index] == 0]
if not failure_samples_in_shap_indices.empty:
    sample_failure_idx = failure_samples_in_shap_indices[0] # Tomar el primer índice
    sample_failure_data_preprocessed = X_test_for_shap.loc[sample_failure_idx]

    print(f"\nGenerando SHAP Force Plot para una instancia de 'Falla Real' (índice original: {sample_failure_idx})...")
    try:
        instance_shap_values_raw = explainer.shap_values(sample_failure_data_preprocessed.to_frame().T)
        
        # --- DEBUG PRINTS para instance_shap_values_raw (Force Plot) ---
        print(f"DEBUG: Tipo de instance_shap_values_raw (para force plot falla): {type(instance_shap_values_raw)}")
        if isinstance(instance_shap_values_raw, list):
            print(f"DEBUG: instance_shap_values_raw es una lista. Longitud: {len(instance_shap_values_raw)}")
            for i, val_array in enumerate(instance_shap_values_raw):
                print(f"DEBUG:   instance_shap_values_raw[{i}] Tipo: {type(val_array)}, Forma: {val_array.shape}, Dim: {val_array.ndim}")
            if len(instance_shap_values_raw) > 0:
                shap_values_for_instance = instance_shap_values_raw[0] # Usar [0] si hay solo un elemento o el principal
            else:
                raise ValueError("La lista de valores SHAP para la instancia está vacía.")
        else:
            shap_values_for_instance = instance_shap_values_raw
        # --- FIN DEBUG PRINTS ---

        if isinstance(shap_values_for_instance, np.ndarray):
            if shap_values_for_instance.ndim > 1 and shap_values_for_instance.shape[0] == 1:
                shap_values_for_instance = shap_values_for_instance.flatten()
            elif shap_values_for_instance.ndim == 0: # Caso de array escalar, convertir a 1D
                shap_values_for_instance = np.array([shap_values_for_instance])
        
        # Ajustar expected_value
        force_plot_expected_value = explainer.expected_value
        if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim > 0:
            if len(explainer.expected_value) > 0:
                force_plot_expected_value = explainer.expected_value[0]
            else:
                force_plot_expected_value = 0.0
                print("Advertencia: explainer.expected_value es un array vacío. Usando 0.0 para force_plot.")

        shap.force_plot(force_plot_expected_value, shap_values_for_instance, sample_failure_data_preprocessed, matplotlib=True, show=False)
        plt.tight_layout()
        force_plot_failure_path = os.path.join(FIGURES_DIR, f'shap_force_plot_real_failure_instance_{sample_failure_idx}.png')
        plt.savefig(force_plot_failure_path)
        plt.close()
        print(f"SHAP Force Plot para 'Falla Real' guardado en: {force_plot_failure_path}")
    except Exception as e:
        print(f"Error al generar/guardar SHAP Force Plot para instancia de falla: {e}", file=sys.stderr)
else:
    print("\nNo se encontró ninguna instancia de Real Falla en la muestra SHAP para generar Force Plot. Considera aumentar N_SHAP_SAMPLES.")

print("\n--- Análisis SHAP Completado ---")
print(f"Revisa las visualizaciones SHAP en: {FIGURES_DIR}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline # Ya no se usa este Pipeline directamente, sino el de imblearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Importa el Pipeline de imblearn

import os
import joblib

print("--- Fase 4: Modelado y Optimización ---")

# --- Carga de Datos Preprocesados ---
try:
    print("Intentando cargar datos preprocesados desde 'data/processed/'...")
    
    # Datos para el modelo base (ya resampleados)
    X_train_resampled = pd.read_csv('data/processed/X_train_resampled.csv')
    y_train_resampled = pd.read_csv('data/processed/y_train_resampled.csv').squeeze() # .squeeze() para Series
    
    # Datos pre-SMOTE para el pipeline de GridSearchCV
    X_train = pd.read_csv('data/processed/X_train_preprocessed.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze() # .squeeze() para Series
    
    X_test = pd.read_csv('data/processed/X_test_preprocessed.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze() # .squeeze() para Series
    
    print("Datos preprocesados cargados exitosamente.\n")
except FileNotFoundError:
    print("Archivos de datos preprocesados no encontrados en 'data/processed/'.")
    print("Asegúrate de ejecutar primero 'scripts/preprocessing/data_preprocessing.py' desde la raíz del proyecto,")
    print("y que guarde 'X_train_preprocessed.csv' y 'y_train.csv' además de los resampleados.")
    exit()

# --- 1. Entrenamiento y Evaluación del Modelo Base (Random Forest) ---
# Este es el modelo RF que ya teníamos, lo mantendremos para comparación
print("## 1. Entrenamiento y Evaluación del Modelo Base (Random Forest)\n")
rf_model_base = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_base.fit(X_train_resampled, y_train_resampled)

y_pred_rf_base = rf_model_base.predict(X_test)
y_pred_proba_rf_base = rf_model_base.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva (Éxito)

print("\n--- Reporte de Clasificación (Random Forest Base) ---")
print(classification_report(y_test, y_pred_rf_base, target_names=['Falla', 'Éxito']))

print("\n--- Matriz de Confusión (Random Forest Base) ---")
cm_rf_base = confusion_matrix(y_test, y_pred_rf_base)
sns.heatmap(cm_rf_base, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho Falla', 'Predicho Éxito'], yticklabels=['Real Falla', 'Real Éxito'])
plt.title('Matriz de Confusión (Random Forest Base)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()

auc_roc_rf_base = roc_auc_score(y_test, y_pred_proba_rf_base)
print(f"\n--- AUC-ROC (Random Forest Base): {auc_roc_rf_base:.4f} ---\n")


# --- 2. Optimización de Hiperparámetros con GridSearchCV para Random Forest ---
print("\n## 2. Optimización de Hiperparámetros con GridSearchCV (Random Forest)\n")
print("Definiendo la grilla de hiperparámetros para RandomForestClassifier con SMOTE en el Pipeline...")

# Define el pipeline que incluirá SMOTE y el clasificador
# Los nombres de los pasos ('smote', 'classifier') son importantes para el param_grid
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)), # Paso 1: Sobremuestreo con SMOTE
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')) # Paso 2: Clasificador
])

# Define la grilla de hiperparámetros para buscar
# Los prefijos 'classifier__' y 'smote__' son necesarios para referenciar los parámetros dentro del pipeline
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False],
    # Opcional: optimizar k_neighbors de SMOTE
    # 'smote__k_neighbors': [3, 5]
}

# Configura GridSearchCV con el pipeline como estimador
grid_search_rf = GridSearchCV(estimator=pipeline, # Ahora el estimador es nuestro pipeline
                              param_grid=param_grid,
                              cv=5,
                              scoring='roc_auc',
                              verbose=2,
                              n_jobs=-1)

print("Iniciando la búsqueda en la grilla de hiperparámetros (esto puede tomar un tiempo)...\n")
# Aquí pasamos los datos originales X_train y y_train (sin SMOTE aplicado previamente)
grid_search_rf.fit(X_train, y_train)

print("\n--- Búsqueda de Hiperparámetros Completada ---")
print(f"Mejores hiperparámetros encontrados: {grid_search_rf.best_params_}")
print(f"Mejor score de AUC-ROC en validación cruzada: {grid_search_rf.best_score_:.4f}")

# --- 3. Evaluación del Modelo Random Forest Optimizado ---
print("\n## 3. Evaluación del Modelo Random Forest Optimizado\n")

rf_model_optimized = grid_search_rf.best_estimator_ # Este es nuestro mejor modelo RF (ahora un pipeline)

y_pred_rf_optimized = rf_model_optimized.predict(X_test)
y_pred_proba_rf_optimized = rf_model_optimized.predict_proba(X_test)[:, 1]

print("\n--- Reporte de Clasificación (Random Forest Optimizado) ---")
print(classification_report(y_test, y_pred_rf_optimized, target_names=['Falla', 'Éxito']))

print("\n--- Matriz de Confusión (Random Forest Optimizado) ---")
cm_rf_optimized = confusion_matrix(y_test, y_pred_rf_optimized)
sns.heatmap(cm_rf_optimized, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho Falla', 'Predicho Éxito'], yticklabels=['Real Falla', 'Real Éxito'])
plt.title('Matriz de Confusión (Random Forest Optimizado)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()

auc_roc_rf_optimized = roc_auc_score(y_test, y_pred_proba_rf_optimized)
print(f"\n--- AUC-ROC (Random Forest Optimizado): {auc_roc_rf_optimized:.4f} ---\n")

# --- Verificación de Distribución de Clases Predichas vs Reales ---
print("\n--- Verificación de Clases Reales en y_test ---")
print(y_test.value_counts())
print("\n--- Verificación de Clases Predichas por el Modelo Optimizado ---")
print(pd.Series(y_pred_rf_optimized).value_counts())

print("\n--- Distribución de Probabilidades Predichas para la clase 'Éxito' ---")
print(pd.Series(y_pred_proba_rf_optimized).head(20)) # Solo mostramos las primeras 20 para no saturar la salida
print(f"Probabilidad mínima predicha para 'Éxito': {np.min(y_pred_proba_rf_optimized):.4f}")
print(f"Probabilidad máxima predicha para 'Éxito': {np.max(y_pred_proba_rf_optimized):.4f}")
print(f"Probabilidad promedio predicha para 'Éxito': {np.mean(y_pred_proba_rf_optimized):.4f}")


# --- 4. Comparación de Modelos: Curvas ROC ---
print("\n## 4. Comparación de Modelos: Curvas ROC\n")

fpr_rf_base, tpr_rf_base, _ = roc_curve(y_test, y_pred_proba_rf_base)
fpr_rf_optimized, tpr_rf_optimized, _ = roc_curve(y_test, y_pred_proba_rf_optimized)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf_base, tpr_rf_base, color='blue', lw=2, label=f'Random Forest Base (AUC = {auc_roc_rf_base:.2f})')
plt.plot(fpr_rf_optimized, tpr_rf_optimized, color='green', lw=2, label=f'Random Forest Optimizado (AUC = {auc_roc_rf_optimized:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
plt.title('Curvas ROC - Comparación de Modelos')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# --- 5. Importancia de Características (Random Forest Optimizado) ---
print("\n## 5. Importancia de Características (Random Forest Optimizado)\n")
# Usamos X_test para obtener los nombres de las columnas preprocesadas,
# ya que rf_model_optimized se entrenó con los mismos features que X_test.
feature_names = X_test.columns

# Asegúrate de que feature_importances_optimized_df no se está utilizando antes de esta sección.
# La importancia de características viene del clasificador DENTRO del pipeline.
importances_optimized = rf_model_optimized.named_steps['classifier'].feature_importances_
feature_importances_optimized_df = pd.DataFrame({'feature': feature_names, 'importance': importances_optimized})
feature_importances_optimized_df = feature_importances_optimized_df.sort_values(by='importance', ascending=False)

print("Importancia de características según Random Forest Optimizado:\n")
print(feature_importances_optimized_df.head(15)) # Mostrar top 15

plt.figure(figsize=(12, 8))
# Reemplaza la advertencia de 'palette' si aparece con 'hue'
# sns.barplot(x='importance', y='feature', data=feature_importances_optimized_df.head(15), palette='viridis', hue='feature', legend=False)
sns.barplot(x='importance', y='feature', data=feature_importances_optimized_df.head(15), palette='viridis') # Mantengo la línea original si funciona para tu matplotlib/seaborn versión
plt.title('Top 15 Características Más Importantes (Random Forest Optimizado)', fontsize=16)
plt.xlabel('Importancia', fontsize=12)
plt.ylabel('Característica', fontsize=12)
plt.show()


# --- Guardar el modelo optimizado (opcional pero recomendado) ---
print("\n--- Guardando el modelo Random Forest optimizado ---")
model_save_path = 'models/random_forest/rf_model_optimized.joblib'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(rf_model_optimized, model_save_path)
print(f"Modelo optimizado guardado en: {model_save_path}\n")

print("\n--- Proceso de Modelado y Optimización Completado ---\n")
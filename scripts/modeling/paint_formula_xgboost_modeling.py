import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, precision_recall_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import os
import joblib

print("--- Fase 4: Modelado y Optimización con XGBoost ---")

# --- Carga de Datos Preprocesados ---
try:
    print("Intentando cargar datos preprocesados desde 'data/processed/'...")
    
    X_train = pd.read_csv('data/processed/X_train_preprocessed.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    
    X_test = pd.read_csv('data/processed/X_test_preprocessed.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print("Datos preprocesados cargados exitosamente.\n")
except FileNotFoundError:
    print("Archivos de datos preprocesados no encontrados en 'data/processed/'.")
    print("Asegúrate de ejecutar primero 'scripts/preprocessing/data_preprocessing.py' desde la raíz del proyecto,")
    print("y que guarde 'X_train_preprocessed.csv' y 'y_train.csv' además de los resampleados.")
    exit()

# --- 1. Configuración de XGBoost para Optimización ---
print("## 1. Configuración de XGBoost para Optimización\n")

# Calcular scale_pos_weight para manejar el desbalanceo
# Si 0 es la clase mayoritaria y 1 es la minoritaria
scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculando scale_pos_weight para XGBoost: {scale_pos_weight_value:.2f}")


# --- 2. Optimización de Hiperparámetros con GridSearchCV para XGBoost ---
print("\n## 2. Optimización de Hiperparámetros con GridSearchCV (XGBoost)\n")
print("Definiendo la grilla de hiperparámetros para XGBoost con SMOTE en el Pipeline...")

pipeline_xgb = ImbPipeline([
    ('smote', SMOTE(random_state=42)), # Paso 1: Sobremuestreo con SMOTE
    ('classifier', xgb.XGBClassifier(random_state=42, 
                                    use_label_encoder=False, # Este parámetro es obsoleto y genera una advertencia, pero no afecta la funcionalidad.
                                    eval_metric='logloss',
                                    scale_pos_weight=scale_pos_weight_value)) # Paso 2: Clasificador XGBoost
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'classifier__gamma': [0, 0.1, 0.2],
    # 'smote__k_neighbors': [3, 5] # Puedes descomentar para optimizar k_neighbors de SMOTE si lo deseas
}

grid_search_xgb = GridSearchCV(estimator=pipeline_xgb,
                               param_grid=param_grid_xgb,
                               cv=5,
                               scoring='roc_auc',
                               verbose=2,
                               n_jobs=-1)

print("Iniciando la búsqueda en la grilla de hiperparámetros (esto puede tomar un tiempo)...\n")
grid_search_xgb.fit(X_train, y_train)

print("\n--- Búsqueda de Hiperparámetros Completada ---")
print(f"Mejores hiperparámetros encontrados: {grid_search_xgb.best_params_}")
print(f"Mejor score de AUC-ROC en validación cruzada: {grid_search_xgb.best_score_:.4f}")

# --- 3. Evaluación del Modelo XGBoost Optimizado con Umbral Por Defecto ---
print("\n## 3. Evaluación del Modelo XGBoost Optimizado (Umbral Por Defecto)\n")

xgb_model_optimized = grid_search_xgb.best_estimator_

y_pred_proba_xgb_optimized = xgb_model_optimized.predict_proba(X_test)[:, 1]
y_pred_xgb_optimized_default = xgb_model_optimized.predict(X_test) # Predicciones con umbral 0.5

print("\n--- Reporte de Clasificación (XGBoost Optimizado - Umbral Por Defecto) ---")
print(classification_report(y_test, y_pred_xgb_optimized_default, target_names=['Falla', 'Éxito']))

print("\n--- Matriz de Confusión (XGBoost Optimizado - Umbral Por Defecto) ---")
cm_xgb_optimized_default = confusion_matrix(y_test, y_pred_xgb_optimized_default)
sns.heatmap(cm_xgb_optimized_default, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho Falla', 'Predicho Éxito'], yticklabels=['Real Falla', 'Real Éxito'])
plt.title('Matriz de Confusión (XGBoost Optimizado - Umbral Por Defecto)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()

auc_roc_xgb_optimized_default = roc_auc_score(y_test, y_pred_proba_xgb_optimized)
print(f"\n--- AUC-ROC (XGBoost Optimizado - Umbral Por Defecto): {auc_roc_xgb_optimized_default:.4f} ---\n")

# --- Verificación de Distribución de Clases Predichas vs Reales ---
print("\n--- Verificación de Clases Reales en y_test ---")
print(y_test.value_counts())
print("\n--- Verificación de Clases Predichas por el Modelo Optimizado (Umbral Por Defecto) ---")
print(pd.Series(y_pred_xgb_optimized_default).value_counts())

print("\n--- Distribución de Probabilidades Predichas para la clase 'Éxito' ---")
print(pd.Series(y_pred_proba_xgb_optimized).head(20))
print(f"Probabilidad mínima predicha para 'Éxito': {np.min(y_pred_proba_xgb_optimized):.4f}")
print(f"Probabilidad máxima predicha para 'Éxito': {np.max(y_pred_proba_xgb_optimized):.4f}")
print(f"Probabilidad promedio predicha para 'Éxito': {np.mean(y_pred_proba_xgb_optimized):.4f}")

# --- 4. Ajuste del Umbral de Clasificación para la Clase 'Éxito' ---
print("\n## 4. Ajuste del Umbral de Clasificación para la Clase 'Éxito'\n")
print("Buscando el umbral óptimo para maximizar el F1-score de la clase 'Éxito'...")

# Calcular precisiones, recalls y umbrales
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_xgb_optimized)

# Calcular F1-scores para cada umbral
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
f1_scores = np.nan_to_num(f1_scores) # Manejar posibles divisiones por cero

# Encontrar el umbral que maximiza el F1-score
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Mejor F1-score para la clase 'Éxito': {f1_scores[optimal_idx]:.4f} en el umbral: {optimal_threshold:.4f}\n")

# Re-evaluar el modelo con el umbral óptimo
y_pred_xgb_optimized_tuned = (y_pred_proba_xgb_optimized >= optimal_threshold).astype(int)

print(f"--- Reporte de Clasificación (XGBoost Optimizado - Umbral Ajustado a {optimal_threshold:.4f}) ---")
print(classification_report(y_test, y_pred_xgb_optimized_tuned, target_names=['Falla', 'Éxito']))

print(f"\n--- Matriz de Confusión (XGBoost Optimizado - Umbral Ajustado a {optimal_threshold:.4f}) ---")
cm_xgb_optimized_tuned = confusion_matrix(y_test, y_pred_xgb_optimized_tuned)
sns.heatmap(cm_xgb_optimized_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho Falla', 'Predicho Éxito'], yticklabels=['Real Falla', 'Real Éxito'])
plt.title(f'Matriz de Confusión (XGBoost Optimizado - Umbral Ajustado a {optimal_threshold:.4f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()

# --- 5. Importancia de Características (XGBoost Optimizado) ---
print("\n## 5. Importancia de Características (XGBoost Optimizado)\n")

importances_xgb = xgb_model_optimized.named_steps['classifier'].get_booster().get_score(importance_type='gain')

feature_names = X_test.columns.tolist()
feature_importance_map = {f'f{i}': name for i, name in enumerate(feature_names)}

feature_importances_xgb_df = pd.DataFrame({
    'feature': [feature_importance_map.get(key, key) for key in importances_xgb.keys()],
    'importance': list(importances_xgb.values())
})
feature_importances_xgb_df = feature_importances_xgb_df.sort_values(by='importance', ascending=False)
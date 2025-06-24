import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Ya se hizo en preprocesamiento
from sklearn.impute import SimpleImputer # Ya se hizo
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Ya se hizo
from imblearn.over_sampling import SMOTE # Ya se hizo

# Importaciones para los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Importaciones para métricas y evaluación
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV # Para futuras mejoras


# --- 0. Cargar Datos Preprocesados (asumiendo que se ejecutó el script de preprocesamiento) ---
# En un entorno real, guardarías los X_train_resampled, y_train_resampled, X_test, y_test
# o pasarías directamente las variables desde el script anterior si lo ejecutas en Jupyter.
# Para este script, simularemos la carga de los datos resultantes del preprocesamiento.

print("--- Fase 4: Modelado de Machine Learning ---")
print("Simulando carga de datos preprocesados...")

# Simulamos que estos vienen del script de preprocesamiento
# En un entorno real, podrías cargarlos desde archivos CSV si los guardaste
# o simplemente asegurarte de que las variables X_train_resampled, y_train_resampled,
# X_test, y_test estén disponibles en tu entorno de ejecución (ej. Jupyter Notebook).

# --- Este bloque de código es solo para que este script sea autónomo si se ejecuta solo ---
# --- En un flujo de trabajo continuo (ej. Jupyter), simplemente usarías las variables ya existentes ---
try:
    # Cargar los datos preprocesados desde la nueva ubicación
    # ¡Importante! La ruta es relativa a la raíz del proyecto si lanzas el script desde la raíz,
    # o relativa a la ubicación del script de modelado si lo lanzas desde esa carpeta.
    # Dado que ahora el script está en scripts/modeling, la ruta a data/processed es '../../data/processed'
    X_train_resampled = pd.read_csv('data/processed/X_train_resampled.csv') 
    y_train_resampled = pd.read_csv('data/processed/y_train_resampled.csv').squeeze() # .squeeze() para Series
    X_test = pd.read_csv('data/processed/X_test_preprocessed.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze() # .squeeze() para Series
    print("Datos preprocesados cargados desde archivos CSV (simulación).\n")
except FileNotFoundError:
    print("Archivos de datos preprocesados no encontrados. Ejecutando el preprocesamiento completo para generar los datos.")
    # Si no se encuentran los archivos, ejecuta el preprocesamiento (copia del código anterior)
    df = pd.read_csv('data/processed/simulated_paint_formulas_with_engineered_features.csv') # para el bloque try/except
    TARGET_COLUMN = 'IsSuccess'
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    imputer_numerical = SimpleImputer(strategy='median')
    X[numerical_cols] = imputer_numerical.fit_transform(X[numerical_cols])

    imputer_categorical = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X_preprocessed = pd.concat([X[numerical_cols], X_encoded_df], axis=1)

    scaler = StandardScaler()
    X_preprocessed[numerical_cols] = scaler.fit_transform(X_preprocessed[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.20, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Preprocesamiento completo ejecutado para generar los datos.\n")

# --- 1. Modelo: Regresión Logística ---
print("## 1. Modelo: Regresión Logística\n")

# 1.1 Inicialización y Entrenamiento del Modelo
# Definición: La Regresión Logística es un clasificador lineal que modela la probabilidad
# de que una instancia dada pertenezca a una clase en particular. Utiliza una función
# logística (sigmoide) para transformar la salida lineal en una probabilidad entre 0 y 1.
# Por qué se hace: Es un excelente modelo de línea base. Es simple, rápido de entrenar
# y sus coeficientes ofrecen cierta interpretabilidad sobre la influencia de las características.
# Funcionamiento estadístico: Estima los pesos (coeficientes) para cada característica
# que maximizan la probabilidad de observar los datos de entrenamiento, utilizando
# el método de máxima verosimilitud. La ecuación es p(y=1|X) = 1 / (1 + exp(-(b0 + b1x1 + ... + bnxn))).
# 'solver' es el algoritmo de optimización, 'liblinear' es bueno para datasets pequeños y medianos.
# 'random_state' asegura la reproducibilidad de los resultados.
log_reg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # 'balanced' puede ayudar, pero SMOTE ya maneja
print("Entrenando el modelo de Regresión Logística...")
log_reg_model.fit(X_train_resampled, y_train_resampled)
print("Entrenamiento de Regresión Logística completado.\n")

# 1.2 Predicciones
# Definición: Usamos el modelo entrenado para predecir las clases y las probabilidades
# en el conjunto de prueba (X_test), que son datos que el modelo nunca ha visto.
# Por qué se hace: Es la forma de evaluar cómo el modelo generaliza a datos nuevos.
# 'predict()' da la clase binaria (0 o 1). 'predict_proba()' da la probabilidad
# de cada clase. Para la clase positiva (1), es la probabilidad de que sea un 'Éxito'.
y_pred_lr = log_reg_model.predict(X_test)
y_pred_proba_lr = log_reg_model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva (Éxito)
print("Predicciones realizadas en el conjunto de prueba.\n")

# 1.3 Evaluación del Modelo de Regresión Logística
print("### 1.3 Evaluación de Regresión Logística\n")

# 1.3.1 Reporte de Clasificación
# Definición: Muestra métricas clave para cada clase (0 y 1).
# Por qué se hace: Es crucial para datasets desbalanceados, ya que la 'accuracy' (precisión general)
# puede ser engañosa. Queremos ver el rendimiento en la clase minoritaria ('Éxito').
# - Precision: TP / (TP + FP) - De todas las veces que el modelo predijo 'Éxito', ¿cuántas fueron correctas?
#   Para un formulador: ¿Cuántas de las fórmulas que el modelo dijo que serían exitosas, realmente lo fueron? Minimizar el desperdicio.
# - Recall (Sensibilidad): TP / (TP + FN) - De todas las fórmulas que realmente fueron 'Éxito', ¿cuántas predijo correctamente el modelo?
#   Para un formulador: ¿Cuántas fórmulas exitosas somos capaces de identificar? No queremos perder oportunidades.
# - F1-score: 2 * (Precision * Recall) / (Precision + Recall) - Es la media armónica de Precision y Recall.
#   Es un buen balance entre ambas métricas, especialmente importante en desbalance.
# - Support: El número de instancias reales de cada clase en el conjunto de prueba.
print("--- Reporte de Clasificación (Regresión Logística) ---")
print(classification_report(y_test, y_pred_lr))
print("\n")

# 1.3.2 Matriz de Confusión
# Definición: Una tabla que resume el rendimiento del modelo de clasificación.
# Por qué se hace: Muestra los conteos de predicciones correctas e incorrectas para cada clase.
# - Verdaderos Positivos (TP): Se predijo 'Éxito' y fue 'Éxito'. (Arriba a la izquierda si el 0 es falla y 1 es éxito)
# - Verdaderos Negativos (TN): Se predijo 'Falla' y fue 'Falla'. (Abajo a la derecha)
# - Falsos Positivos (FP): Se predijo 'Éxito' pero fue 'Falla'. (Error Tipo I - 'False Alarm')
#   Para un formulador: Desperdicio de recursos al intentar escalar una fórmula que no funcionará.
# - Falsos Negativos (FN): Se predijo 'Falla' pero fue 'Éxito'. (Error Tipo II - 'Missed Opportunity')
#   Para un formulador: No identificamos una fórmula exitosa, perdiendo la oportunidad de un gran producto. ¡A menudo es el error más costoso!
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("--- Matriz de Confusión (Regresión Logística) ---")
print(cm_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicción: Falla', 'Predicción: Éxito'],
            yticklabels=['Real: Falla', 'Real: Éxito'])
plt.title('Matriz de Confusión - Regresión Logística', fontsize=14)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.ylabel('Etiqueta Verdadera', fontsize=12)
plt.show()
print("\n")

# 1.3.3 Curva ROC y AUC-ROC
# Definición: La Curva Característica Operativa del Receptor (ROC) grafica la tasa de
# verdaderos positivos (TPR/Recall) contra la tasa de falsos positivos (FPR) en varios
# umbrales de clasificación. El Área Bajo la Curva (AUC) es una métrica agregada
# que representa la capacidad de discriminación del modelo entre clases.
# Por qué se hace: Es una excelente métrica para datasets desbalanceados porque
# no es sensible al desbalance de clases ni al umbral de clasificación elegido.
# Un AUC de 0.5 es aleatorio, 1.0 es perfecto. Cuanto más cerca esté la curva
# de la esquina superior izquierda, mejor el modelo.
# Funcionamiento estadístico: La TPR es la probabilidad de detectar un verdadero positivo
# cuando realmente lo es. La FPR es la probabilidad de detectar un falso positivo cuando
# en realidad es un negativo. La curva ROC muestra el trade-off entre estos dos.
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

print(f"--- AUC-ROC (Regresión Logística): {roc_auc_lr:.4f} ---\n")

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Recall')
plt.title('Curva ROC - Regresión Logística', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print("\n")


# --- 2. Modelo: Random Forest Classifier ---
print("## 2. Modelo: Random Forest Classifier\n")

# 2.1 Inicialización y Entrenamiento del Modelo
# Definición: Un ensamble de múltiples árboles de decisión. Cada árbol se entrena
# en una muestra aleatoria de los datos y con un subconjunto aleatorio de características.
# Las predicciones finales son por voto mayoritario (clasificación) o promedio (regresión).
# Por qué se hace: Muy robusto, maneja bien datos no lineales y complejas interacciones
# entre características, y es menos propenso al sobreajuste que un solo árbol.
# 'n_estimators': Número de árboles en el bosque (más árboles generalmente mejor, pero más lento).
# 'random_state': Para reproducibilidad.
# 'class_weight': Aunque usamos SMOTE, 'balanced' aquí puede refinar el tratamiento del desbalance.
# 'max_depth': Profundidad máxima de cada árbol, ayuda a controlar el sobreajuste.
# 'min_samples_leaf': Número mínimo de muestras requeridas para estar en un nodo hoja.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10, min_samples_leaf=5)
print("Entrenando el modelo Random Forest...")
rf_model.fit(X_train_resampled, y_train_resampled)
print("Entrenamiento de Random Forest completado.\n")

# 2.2 Predicciones
# Se hacen las predicciones de clase y de probabilidad de forma similar a la Regresión Logística.
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
print("Predicciones realizadas en el conjunto de prueba.\n")

# 2.3 Evaluación del Modelo de Random Forest
print("### 2.3 Evaluación de Random Forest\n")

# 2.3.1 Reporte de Clasificación (Random Forest)
print("--- Reporte de Clasificación (Random Forest) ---")
print(classification_report(y_test, y_pred_rf))
print("\n")

# 2.3.2 Matriz de Confusión (Random Forest)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("--- Matriz de Confusión (Random Forest) ---")
print(cm_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicción: Falla', 'Predicción: Éxito'],
            yticklabels=['Real: Falla', 'Real: Éxito'])
plt.title('Matriz de Confusión - Random Forest', fontsize=14)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.ylabel('Etiqueta Verdadera', fontsize=12)
plt.show()
print("\n")

# 2.3.3 Curva ROC y AUC-ROC (Random Forest)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

print(f"--- AUC-ROC (Random Forest): {roc_auc_rf:.4f} ---\n")

plt.figure(figsize=(7, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Recall')
plt.title('Curva ROC - Random Forest', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print("\n")


# --- 3. Comparación de Curvas ROC ---
# Definición: Graficar ambas curvas ROC en el mismo gráfico para una comparación visual directa.
# Por qué se hace: Permite ver qué modelo tiene una mejor capacidad de discriminación general.
# La curva que envuelve más área y se acerca más a la esquina superior izquierda es mejor.
print("## 3. Comparación de Modelos: Curvas ROC\n")
plt.figure(figsize=(8, 7))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Regresión Logística (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Recall')
plt.title('Comparación de Curvas ROC de Modelos', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# --- 4. Importancia de Características (Solo para Random Forest) ---
# Definición: Random Forest puede estimar la importancia de cada característica en la toma de decisiones.
# Por qué se hace: Es invaluable para la interpretabilidad del modelo y para la toma de decisiones
# en I+D. Permite al formulador entender qué aspectos de la fórmula son más críticos
# para determinar el éxito.
# Funcionamiento estadístico: La importancia se calcula en base a la reducción de la impureza
# (como Gini impurity o entropía) que cada característica aporta a lo largo de todos los árboles
# del bosque. Las características que consistentemente reducen más la impureza en los nodos
# de los árboles son consideradas más importantes.
print("## 4. Importancia de Características (Random Forest)\n")
feature_names = X_test.columns # Usamos X_test para obtener los nombres de las columnas preprocesadas
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("--- Top 10 Características Más Importantes (Random Forest) ---")
print(importance_df.head(10))

plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Importancia de Características (Random Forest)', fontsize=16)
plt.xlabel('Importancia', fontsize=12)
plt.ylabel('Característica', fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- Fase 4: Modelado de Machine Learning Completado ---")
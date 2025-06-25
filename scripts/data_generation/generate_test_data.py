import pandas as pd
import numpy as np
import os

# --- Configuración de la generación de datos de prueba ---
num_test_samples = 10 # Número de filas (fórmulas) a generar en el archivo de prueba
output_dir = 'C:/Users/Víctor/Documents/paint_predictor_mvp/data/test_data' # Nuevo directorio para datos de prueba
output_file = 'paint_formulas_test_batch.csv' # Nombre del archivo CSV de prueba

# Asegurarse de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Definición de columnas y sus rangos/opciones (DEBE COINCIDIR EXACTAMENTE CON EL ENTRENAMIENTO)
# Esta lista debe ser idéntica a la que usaste en generate_simulated_data.py,
# PERO EXCLUYENDO 'IsSuccess' porque este es un archivo para PREDECIR.
data_specs = {
    # Componentes de la fórmula
    "ResinPercentage": {"type": "float", "range": (10.0, 60.0)},
    "PigmentPercentage": {"type": "float", "range": (5.0, 40.0)},
    "SolventPercentage": {"type": "float", "range": (10.0, 70.0)},
    "AdditivePercentage": {"type": "float", "range": (0.5, 10.0)},
    "TotalAdditivesPercentage": {"type": "float", "range": (1.0, 15.0)},

    # Condiciones y propiedades físicas
    "SubstrateType": {"type": "categorical", "options": ["Metal", "Madera", "Plastico", "Concreto", "Fibra"]},
    "ApplicationMethod": {"type": "categorical", "options": ["Brocha", "Rodillo", "Pistola", "Inmersion"]},
    "ApplicationTemp_C": {"type": "float", "range": (5.0, 40.0)},
    "Humidity": {"type": "float", "range": (30.0, 90.0)},
    "PHLevel": {"type": "float", "range": (5.0, 9.0)},
    "Viscosity": {"type": "float", "range": (50.0, 2000.0)},
    "DryingTime_Hours": {"type": "float", "range": (0.1, 5.0)},
    "Coverage": {"type": "float", "range": (5.0, 20.0)},
    "Gloss": {"type": "float", "range": (0.0, 100.0)},
    "HidingPower": {"type": "categorical", "options": ["High", "Medium", "Low"]},

    # Atributos específicos / Binarias (0.0 o 1.0)
    "AcrylicOnWood": {"type": "binary"},
    "EpoxyOnMetal": {"type": "binary"},
    "HighDryingTime": {"type": "binary"},
    "LowApplicationTemp": {"type": "binary"},
    "TiO2OnConcrete": {"type": "binary"},

    # Información de proveedores/tipos (categóricas)
    "Biocide_Supplier": {"type": "categorical", "options": ["SupplierA", "SupplierB", "SupplierC", "SupplierD"]},
    "Coalescent_Supplier": {"type": "categorical", "options": ["SupplierE", "SupplierF", "SupplierG", "SupplierH"]},
    "Defoamer_Supplier": {"type": "categorical", "options": ["SupplierI", "SupplierJ", "SupplierK", "SupplierM"]},
    "Dispersant_Supplier": {"type": "categorical", "options": ["SupplierN", "SupplierO", "SupplierP", "SupplierQ"]},
    "PigmentSupplier": {"type": "categorical", "options": ["SupplierR", "SupplierS", "SupplierT"]},
    "PigmentType": {"type": "categorical", "options": ["TiO2", "CalciumCarbonate", "Talc", "Silica"]},
    "ResinSupplier": {"type": "categorical", "options": ["SupplierU", "SupplierV", "SupplierW"]},
    "ResinType": {"type": "categorical", "options": ["Acrylic", "Epoxy", "Polyurethane", "Alkyd"]},
    "SolventSupplier": {"type": "categorical", "options": ["SupplierX", "SupplierY", "SupplierZ"]},
    "SolventType": {"type": "categorical", "options": ["Water", "Organic", "Mineral Spirits"]},
    "Surfactant_Supplier": {"type": "categorical", "options": ["SupplierAA", "SupplierBB", "SupplierCC"]},
    "Thickener_Supplier": {"type": "categorical", "options": ["SupplierDD", "SupplierEE", "SupplierFF"]},

    # Porcentajes de aditivos específicos (flotantes)
    "Biocide_Percentage": {"type": "float", "range": (0.01, 0.5)},
    "Coalescent_Percentage": {"type": "float", "range": (0.1, 1.0)},
    "Defoamer_Percentage": {"type": "float", "range": (0.05, 0.5)},
    "Dispersant_Percentage": {"type": "float", "range": (0.1, 0.8)},
    "Surfactant_Percentage": {"type": "float", "range": (0.05, 0.6)},
    "Thickener_Percentage": {"type": "float", "range": (0.1, 1.5)},

    # Relaciones calculadas
    "EstimatedDensity": {"type": "float", "range": (0.9, 1.5)},
    "ResinToPigmentRatio": {"type": "float", "range": (0.5, 3.0)},
    "ResinToSolventRatio": {"type": "float", "range": (0.2, 2.0)},
    "PigmentToSolventRatio": {"type": "float", "range": (0.1, 1.0)},
}

# Crear un DataFrame vacío para almacenar los datos
df_test = pd.DataFrame()

# Generar datos para cada columna
print(f"Generando {num_test_samples} muestras de datos de prueba...")
for col_name, specs in data_specs.items():
    if specs["type"] == "float":
        df_test[col_name] = np.random.uniform(specs["range"][0], specs["range"][1], num_test_samples).round(2)
    elif specs["type"] == "categorical":
        df_test[col_name] = np.random.choice(specs["options"], num_test_samples)
    elif specs["type"] == "binary":
        df_test[col_name] = np.random.choice([0.0, 1.0], num_test_samples) # Usar floats para binarias como en el entrenamiento

# Guardar el DataFrame en un archivo CSV
try:
    df_test.to_csv(output_path, index=False)
    print(f"CSV de datos de prueba simulados generado exitosamente en: {output_path}")
    print(f"Dimensiones del DataFrame de prueba generado: {df_test.shape}")
    print("Primeras 5 filas del CSV de prueba generado:")
    print(df_test.head())
    print("\nTipos de datos de las columnas en el CSV de prueba generado:")
    print(df_test.dtypes)
except Exception as e:
    print(f"ERROR: Fallo al guardar el CSV de prueba: {e}")
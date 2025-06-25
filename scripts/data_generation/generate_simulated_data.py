import pandas as pd
import numpy as np
import os

# --- Configuración de la generación de datos ---
num_samples = 15000 # Número total de filas (fórmulas) a generar
output_dir = 'C:/Users/Víctor/Documents/paint_predictor_mvp/data/processed' # Directorio donde se guardará el CSV
output_file = 'simulated_paint_formulas_with_engineered_features.csv' # Nombre del archivo CSV

# Asegurarse de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Definición de columnas y sus rangos/opciones (ajustados para incluir las 5 faltantes)
data_specs = {
    # Componentes de la fórmula
    "ResinPercentage": {"type": "float", "range": (10.0, 60.0)},
    "PigmentPercentage": {"type": "float", "range": (5.0, 40.0)},
    "SolventPercentage": {"type": "float", "range": (10.0, 70.0)},
    "AdditivePercentage": {"type": "float", "range": (0.5, 10.0)}, # <-- ANTES FALTABA
    "TotalAdditivesPercentage": {"type": "float", "range": (1.0, 15.0)}, # Nueva columna, asumo que la tenías en mente

    # Condiciones y propiedades físicas
    "SubstrateType": {"type": "categorical", "options": ["Metal", "Madera", "Plastico", "Concreto", "Fibra"]},
    "ApplicationMethod": {"type": "categorical", "options": ["Brocha", "Rodillo", "Pistola", "Inmersion"]}, # <-- ANTES FALTABA
    "ApplicationTemp_C": {"type": "float", "range": (5.0, 40.0)},
    "Humidity": {"type": "float", "range": (30.0, 90.0)}, # <-- ANTES FALTABA
    "PHLevel": {"type": "float", "range": (5.0, 9.0)}, # <-- ANTES FALTABA
    "Viscosity": {"type": "float", "range": (50.0, 2000.0)},
    "DryingTime_Hours": {"type": "float", "range": (0.1, 5.0)},
    "Coverage": {"type": "float", "range": (5.0, 20.0)}, # <-- ANTES FALTABA
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

    # Columna objetivo
    "IsSuccess": {"type": "binary"} # 0 para fallo, 1 para éxito
}

# Crear un DataFrame vacío para almacenar los datos
df = pd.DataFrame()

# Generar datos para cada columna
print(f"Generando {num_samples} muestras de datos...")
for col_name, specs in data_specs.items():
    if specs["type"] == "float":
        df[col_name] = np.random.uniform(specs["range"][0], specs["range"][1], num_samples).round(2)
    elif specs["type"] == "categorical":
        df[col_name] = np.random.choice(specs["options"], num_samples)
    elif specs["type"] == "binary":
        df[col_name] = np.random.choice([0, 1], num_samples) # Para 'IsSuccess' y binarias

# Asegurarse de que los tipos de datos en el DataFrame coincidan con los esperados en el training script
# Esto es más para la verificación interna de este script de generación,
# ya que el training script ahora usa 'dtype' en pd.read_csv
for col_name, specs in data_specs.items():
    if specs["type"] == "float" or specs["type"] == "binary":
        df[col_name] = df[col_name].astype(float)
    elif specs["type"] == "categorical":
        df[col_name] = df[col_name].astype(str)

# Guardar el DataFrame en un archivo CSV
try:
    df.to_csv(output_path, index=False)
    print(f"CSV de datos simulados generado exitosamente en: {output_path}")
    print(f"Dimensiones del DataFrame generado: {df.shape}")
    print("Primeras 5 filas del CSV generado:")
    print(df.head())
    print("\nTipos de datos de las columnas en el CSV generado:")
    print(df.dtypes)
except Exception as e:
    print(f"ERROR: Fallo al guardar el CSV: {e}")
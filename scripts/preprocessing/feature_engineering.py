import pandas as pd
import numpy as np
import os

print("--- Fase 2: Ingeniería de Características ---")

# --- Carga de Datos ---
try:
    print("Intentando cargar datos brutos desde 'data/raw/simulated_paint_formulas_complex.csv'...")
    df = pd.read_csv('data/raw/simulated_paint_formulas_complex.csv')
    print("Datos brutos cargados exitosamente.\n")
except FileNotFoundError:
    print("Archivo 'simulated_paint_formulas_complex.csv' no encontrado en 'data/raw/'.")
    print("Asegúrate de ejecutar primero 'scripts/data_generation/simulate_complex_paint_formulas.py'.")
    exit()

# --- Ingeniería de Características ---
print("Aplicando lógicas de ingeniería de características...")

# 1. Ratios de Componentes Principales
# Justificación: Las proporciones relativas de los componentes principales (Resina, Pigmento, Solvente)
# son a menudo más importantes que sus porcentajes absolutos, ya que determinan la composición central de la pintura.
# Son una forma de normalizar la composición.
df['ResinToPigmentRatio'] = df['ResinPercentage'] / (df['PigmentPercentage'] + 1e-6) # Evitar división por cero
df['ResinToSolventRatio'] = df['ResinPercentage'] / (df['SolventPercentage'] + 1e-6)
df['PigmentToSolventRatio'] = df['PigmentPercentage'] / (df['SolventPercentage'] + 1e-6)
print("- Creados ratios de componentes principales.")

# 2. Total de Aditivos
# Justificación: El porcentaje combinado de todos los aditivos podría ser una característica importante,
# ya que un exceso o defecto general de aditivos podría influir en el rendimiento.
df['TotalAdditivesPercentage'] = df['Dispersant_Percentage'] + \
                                 df['Thickener_Percentage'] + \
                                 df['Defoamer_Percentage'] + \
                                 df['Coalescent_Percentage'] + \
                                 df['Biocide_Percentage'] + \
                                 df['Surfactant_Percentage']
print("- Creado porcentaje total de aditivos.")

# 3. Interacciones entre Tipo de Sustrato y Tipos de Componentes
# Justificación: El rendimiento de una pintura puede depender fuertemente de la combinación
# del tipo de resina o pigmento con el sustrato al que se aplica. Por ejemplo, una resina
# acrílica se comporta diferente en madera que en metal.
# Creamos características binarias para combinaciones específicas que podrían ser relevantes.
# Se pueden añadir más según el conocimiento del dominio.
df['AcrylicOnWood'] = ((df['ResinType'] == 'Acrylic') & (df['SubstrateType'] == 'Wood')).astype(int)
df['EpoxyOnMetal'] = ((df['ResinType'] == 'Epoxy') & (df['SubstrateType'] == 'Metal')).astype(int)
df['TiO2OnConcrete'] = ((df['PigmentType'] == 'TiO2') & (df['SubstrateType'] == 'Concrete')).astype(int)
print("- Creadas interacciones entre tipo de sustrato y tipos de componentes.")

# 4. Densidad Estimada Simple (característica compuesta)
# Justificación: Una estimación simplificada de la densidad podría influir en la aplicación y el rendimiento.
# Esto es una suposición simplificada de densidades típicas.
# (Más avanzado: requeriría densidades reales de cada componente, pero para simulación es útil)
density_map = {
    'Resin': 1.1, 'Pigment': 3.0, 'Solvent': 0.9, # Valores promedio
    'Dispersant': 1.05, 'Thickener': 1.0, 'Defoamer': 0.95,
    'Coalescent': 0.98, 'Biocide': 1.0, 'Surfactant': 1.0
}
df['EstimatedDensity'] = (
    df['ResinPercentage'] * density_map['Resin'] +
    df['PigmentPercentage'] * density_map['Pigment'] +
    df['SolventPercentage'] * density_map['Solvent'] +
    df['Dispersant_Percentage'] * density_map['Dispersant'] +
    df['Thickener_Percentage'] * density_map['Thickener'] +
    df['Defoamer_Percentage'] * density_map['Defoamer'] +
    df['Coalescent_Percentage'] * density_map['Coalescent'] +
    df['Biocide_Percentage'] * density_map['Biocide'] +
    df['Surfactant_Percentage'] * density_map['Surfactant']
) / 100 # Dividir por 100 si los porcentajes suman 100
print("- Creada característica de densidad estimada.")

# 5. Indicadores de Condiciones de Aplicación Extremas
# Justificación: Temperaturas o tiempos de secado muy altos/bajos pueden indicar condiciones
# desafiantes que impactan el éxito de la pintura.
df['HighDryingTime'] = (df['DryingTime_Hours'] > 10).astype(int)
df['LowApplicationTemp'] = (df['ApplicationTemp_C'] < 15).astype(int)
print("- Creados indicadores de condiciones de aplicación extremas.")

# --- Guardar el Dataset con Nuevas Características ---
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# Guardamos el dataset con las nuevas características para el siguiente paso (preprocesamiento final)
output_path = os.path.join(output_dir, 'simulated_paint_formulas_with_engineered_features.csv')
df.to_csv(output_path, index=False)
print(f"\nDataset con ingeniería de características guardado exitosamente en: {output_path}")

print("\n--- Ingeniería de Características Completada ---")
import pandas as pd
import numpy as np
import os

print("--- Fase 2: Ingeniería de Características ---")

def apply_feature_engineering(df):
    """
    Aplica ingeniería de características al DataFrame de fórmulas de pintura.
    """
    # Crear una copia para evitar SettingWithCopyWarning
    df_copy = df.copy()

    # 1. Ratios de Componentes Principales
    df_copy['ResinToPigmentRatio'] = df_copy['ResinPercentage'] / (df_copy['PigmentPercentage'] + 1e-6)
    df_copy['ResinToSolventRatio'] = df_copy['ResinPercentage'] / (df_copy['SolventPercentage'] + 1e-6)
    df_copy['PigmentToSolventRatio'] = df_copy['PigmentPercentage'] / (df_copy['SolventPercentage'] + 1e-6)

    # 2. Total de Aditivos
    df_copy['TotalAdditivesPercentage'] = df_copy['Dispersant_Percentage'] + \
                                         df_copy['Thickener_Percentage'] + \
                                         df_copy['Defoamer_Percentage'] + \
                                         df_copy['Coalescent_Percentage'] + \
                                         df_copy['Biocide_Percentage'] + \
                                         df_copy['Surfactant_Percentage']

    # 3. Interacciones entre Tipo de Sustrato y Tipos de Componentes
    df_copy['AcrylicOnWood'] = ((df_copy['ResinType'] == 'Acrylic') & (df_copy['SubstrateType'] == 'Wood')).astype(int)
    df_copy['EpoxyOnMetal'] = ((df_copy['ResinType'] == 'Epoxy') & (df_copy['SubstrateType'] == 'Metal')).astype(int)
    df_copy['TiO2OnConcrete'] = ((df_copy['PigmentType'] == 'TiO2') & (df_copy['SubstrateType'] == 'Concrete')).astype(int)

    # 4. Densidad Estimada Simple
    density_map = {
        'Resin': 1.1, 'Pigment': 3.0, 'Solvent': 0.9,
        'Dispersant': 1.05, 'Thickener': 1.0, 'Defoamer': 0.95,
        'Coalescent': 0.98, 'Biocide': 1.0, 'Surfactant': 1.0
    }
    df_copy['EstimatedDensity'] = (
        df_copy['ResinPercentage'] * density_map['Resin'] +
        df_copy['PigmentPercentage'] * density_map['Pigment'] +
        df_copy['SolventPercentage'] * density_map['Solvent'] +
        df_copy['Dispersant_Percentage'] * density_map['Dispersant'] +
        df_copy['Thickener_Percentage'] * density_map['Thickener'] +
        df_copy['Defoamer_Percentage'] * density_map['Defoamer'] +
        df_copy['Coalescent_Percentage'] * density_map['Coalescent'] +
        df_copy['Biocide_Percentage'] * density_map['Biocide'] +
        df_copy['Surfactant_Percentage'] * density_map['Surfactant']
    ) / 100 

    # 5. Indicadores de Condiciones de Aplicación Extremas
    df_copy['HighDryingTime'] = (df_copy['DryingTime_Hours'] > 10).astype(int)
    df_copy['LowApplicationTemp'] = (df_copy['ApplicationTemp_C'] < 15).astype(int)

    return df_copy

if __name__ == "__main__":
    input_dir = 'data/raw'
    output_dir = 'data/processed'
    input_file = 'simulated_paint_formulas_complex.csv'
    output_file = 'processed_paint_formulas_complex.csv'

    # Crea el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Construye la ruta completa al archivo de entrada
    # USAMOS os.path.normpath para normalizar la ruta y asegurar que las barras sean correctas
    input_path = os.path.normpath(os.path.join(input_dir, input_file))
    output_path = os.path.normpath(os.path.join(output_dir, output_file))

    print(f"Intentando cargar datos brutos desde '{input_path}'...")
    try:
        raw_df = pd.read_csv(input_path)
        print(f"Datos brutos cargados exitosamente. Forma: {raw_df.shape}")
        
        # Aplicar ingeniería de características
        processed_df = apply_feature_engineering(raw_df)
        print(f"Ingeniería de características aplicada. Forma: {processed_df.shape}")

        # Guardar el DataFrame procesado
        processed_df.to_csv(output_path, index=False)
        print(f"Datos procesados guardados exitosamente en: {output_path}")

    except FileNotFoundError:
        print(f"Error: Archivo '{input_file}' no encontrado en '{input_dir}/'.")
        print("Asegúrate de ejecutar primero 'scripts/data_generation/simulate_complex_paint_formulas.py'.")
    except Exception as e:
        print(f"Ocurrió un error al procesar los datos: {e}")

print("--- Ingeniería de Características Completada ---")
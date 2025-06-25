import pandas as pd
import numpy as np
import os

print("--- Fase 1: Generación de Datos de Fórmulas de Pintura Más Completos ---")

def generate_complex_paint_formulas(num_samples=10000): # Aumentamos el número de muestras
    """
    Genera un dataset simulado de fórmulas de pintura más complejo y completo.
    Añade más categorías, rangos variados y una lógica de éxito más intrincada.
    """
    np.random.seed(42) # Para reproducibilidad

    # Definiciones de rangos y categorías (ampliadas)
    resin_types = ['Acrylic', 'Epoxy', 'Polyurethane', 'Alkyd', 'Latex', 'Silicone']
    resin_suppliers = ['SupplierA', 'SupplierB', 'SupplierC', 'SupplierD_Resin', 'SupplierE_Resin']
    
    pigment_types = ['TiO2', 'Iron Oxide', 'Carbon Black', 'Phthalo Blue', 'Quinacridone', 'Metallic']
    pigment_suppliers = ['SupplierF', 'SupplierG_Pigment', 'SupplierH_Pigment', 'SupplierI_Pigment', 'SupplierJ_Pigment', 'SupplierK_Pigment', 'SupplierL_Pigment']
    
    solvent_types = ['Water', 'Mineral Spirits', 'Acetone', 'Xylene', 'MEK', 'Glycol Ether']
    solvent_suppliers = ['SupplierM', 'SupplierN', 'SupplierO_Solvent', 'SupplierP_Solvent']

    dispersant_suppliers = ['SupplierQ_Add', 'SupplierR_Add', 'SupplierS_Add']
    thickener_suppliers = ['SupplierT_Add', 'SupplierU_Add']
    defoamer_suppliers = ['SupplierV_Add', 'SupplierW_Add']
    coalescent_suppliers = ['SupplierX_Add', 'SupplierY_Add']
    biocide_suppliers = ['SupplierZ_Add', 'SupplierAA_Add']
    surfactant_suppliers = ['SupplierBB_Add', 'SupplierCC_Add']
    
    application_methods = ['Brocha', 'Rodillo', 'Pistola', 'Inmersion']
    substrate_types = ['Wood', 'Metal', 'Plastic', 'Concrete', 'Drywall']

    # Rangos para porcentajes y propiedades (más variados)
    data = {
        'ResinType': np.random.choice(resin_types, num_samples),
        'ResinSupplier': np.random.choice(resin_suppliers, num_samples),
        'ResinPercentage': np.random.uniform(20, 60, num_samples),
        
        'PigmentType': np.random.choice(pigment_types, num_samples),
        'PigmentSupplier': np.random.choice(pigment_suppliers, num_samples),
        'PigmentPercentage': np.random.uniform(5, 40, num_samples),
        
        'SolventType': np.random.choice(solvent_types, num_samples),
        'SolventSupplier': np.random.choice(solvent_suppliers, num_samples),
        'SolventPercentage': np.random.uniform(10, 50, num_samples),

        'Dispersant_Percentage': np.random.uniform(0.1, 2.0, num_samples),
        'Dispersant_Supplier': np.random.choice(dispersant_suppliers, num_samples),
        
        'Thickener_Percentage': np.random.uniform(0.05, 1.5, num_samples),
        'Thickener_Supplier': np.random.choice(thickener_suppliers, num_samples),
        
        'Defoamer_Percentage': np.random.uniform(0.01, 1.0, num_samples),
        'Defoamer_Supplier': np.random.choice(defoamer_suppliers, num_samples),

        'Coalescent_Percentage': np.random.uniform(0.5, 5.0, num_samples),
        'Coalescent_Supplier': np.random.choice(coalescent_suppliers, num_samples),
        
        'Biocide_Percentage': np.random.uniform(0.01, 0.5, num_samples),
        'Biocide_Supplier': np.random.choice(biocide_suppliers, num_samples),

        'Surfactant_Percentage': np.random.uniform(0.1, 3.0, num_samples),
        'Surfactant_Supplier': np.random.choice(surfactant_suppliers, num_samples),
        
        'Gloss': np.random.uniform(30, 95, num_samples),
        'Viscosity': np.random.uniform(500, 5000, num_samples),
        'DryingTime_Hours': np.random.uniform(1, 12, num_samples),
        'ApplicationTemp_C': np.random.uniform(10, 35, num_samples),
        'SubstrateType': np.random.choice(substrate_types, num_samples),
        
        'AdditivePercentage': np.random.uniform(1, 15, num_samples), # Este es un porcentaje general, no la suma de los específicos.
        'PHLevel': np.random.uniform(6.0, 9.0, num_samples),
        'Coverage': np.random.uniform(5.0, 20.0, num_samples),
        'Humidity': np.random.uniform(30.0, 90.0, num_samples),
        'ApplicationMethod': np.random.choice(application_methods, num_samples),
    }

    df = pd.DataFrame(data)

    # Convertir HidingPower numérico a categórico para consistencia con train_simple_model.py
    hiding_power_num = np.random.uniform(0.7, 1.0, num_samples)
    df['HidingPower'] = pd.cut(hiding_power_num, 
                               bins=[0.0, 0.75, 0.85, 1.00001], # Ajuste para incluir 1.0 en 'Alto'
                               labels=['Bajo', 'Medio', 'Alto'], 
                               right=False, 
                               include_lowest=True)
    df['HidingPower'] = df['HidingPower'].astype(str)
    if df['HidingPower'].isnull().any():
        print("ADVERTENCIA: NaN en HidingPower después de la categorización. Revisar rangos o bins.")
        # Opcional: imputar NaNs si aparecen (ej. df['HidingPower'].fillna('Medio', inplace=True))


    # Asegurarse de que los porcentajes sumen aproximadamente 100% para los 3 componentes principales
    # Esto es una normalización para que sumen ~100%, luego una ligera variación para simular ruido.
    total_base_perc = df['ResinPercentage'] + df['PigmentPercentage'] + df['SolventPercentage']
    df['ResinPercentage'] = (df['ResinPercentage'] / total_base_perc) * np.random.uniform(90, 110, num_samples)
    df['PigmentPercentage'] = (df['PigmentPercentage'] / total_base_perc) * np.random.uniform(90, 110, num_samples)
    df['SolventPercentage'] = (df['SolventPercentage'] / total_base_perc) * np.random.uniform(90, 110, num_samples)
    
    # Simular 'IsSuccess' con una lógica más compleja y basada en interacciones.
    df['IsSuccess'] = 0

    success_condition1 = (df['ResinPercentage'].between(35, 55)) & \
                         (df['PigmentPercentage'].between(15, 30)) & \
                         (df['SolventPercentage'].between(20, 40)) & \
                         (df['Viscosity'].between(1000, 3000)) & \
                         (df['HidingPower'] == 'Alto')

    success_condition2 = ((df['ResinType'] == 'Acrylic') & (df['SolventType'] == 'Water')) | \
                         ((df['ResinType'] == 'Epoxy') & (df['SolventType'] == 'Xylene')) | \
                         ((df['ResinType'] == 'Polyurethane') & (df['SolventType'] == 'MEK'))

    success_condition3 = (df['Dispersant_Percentage'] < 1.0) & \
                         (df['Thickener_Percentage'] < 0.5) & \
                         (df['Defoamer_Percentage'] < 0.2) & \
                         (df['PHLevel'].between(6.5, 8.5)) # Usa la nueva columna PHLevel

    success_condition4 = (df['DryingTime_Hours'] < 6) & \
                         (df['ApplicationTemp_C'] > 18) & \
                         (df['Humidity'] < 75) & \
                         (df['Coverage'] > 12) & \
                         (df['ApplicationMethod'] == 'Pistola') # Usa las nuevas columnas

    # Asigna éxito con probabilidades basadas en las condiciones
    df.loc[success_condition1 & (np.random.rand(len(df)) < 0.8), 'IsSuccess'] = 1 
    df.loc[success_condition2 & (np.random.rand(len(df)) < 0.7), 'IsSuccess'] = 1
    df.loc[success_condition3 & (np.random.rand(len(df)) < 0.6), 'IsSuccess'] = 1
    df.loc[success_condition4 & (np.random.rand(len(df)) < 0.75), 'IsSuccess'] = 1

    # Reglas más estrictas para garantizar algunos éxitos claros y algunas fallas claras
    df.loc[(df['ResinType'] == 'Silicone') & (df['SolventType'] == 'Water') & (df['Gloss'] < 50), 'IsSuccess'] = 0
    df.loc[(df['HidingPower'] == 'Bajo') & (df['PigmentPercentage'] < 10), 'IsSuccess'] = 0
    df.loc[(df['Viscosity'] > 4500) | (df['Viscosity'] < 600), 'IsSuccess'] = 0
    df.loc[(df['ResinType'] == 'Acrylic') & (df['PigmentType'] == 'TiO2') & (df['HidingPower'] == 'Alto'), 'IsSuccess'] = 1

    # Asegurar un balance mínimo de 'Éxito'
    num_success = df['IsSuccess'].sum()
    target_success_count = int(num_samples * 0.20) # Apunta a al menos 20% de éxitos
    
    if num_success < target_success_count:
        fail_indices = df[df['IsSuccess'] == 0].index
        num_to_convert = target_success_count - num_success
        
        if num_to_convert > 0 and len(fail_indices) > 0:
            # Seleccionar aleatoriamente entre las fallas para convertir a éxito
            convert_indices = np.random.choice(fail_indices, min(num_to_convert, len(fail_indices)), replace=False)
            df.loc[convert_indices, 'IsSuccess'] = 1
            print(f"Ajustado: Convertidas {len(convert_indices)} fórmulas a 'Éxito' para un mejor balanceo simulado.")
        else:
            print("No se pudo ajustar el balance de 'Éxito' debido a la falta de muestras para convertir.")

    print(f"Dataset simulado generado con {num_samples} muestras.")
    print(f"Distribución de 'IsSuccess':\n{df['IsSuccess'].value_counts(normalize=True)}")

    return df

# --- Generación y Guardado de la Base de Datos ---
if __name__ == "__main__":
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerando una nueva base de datos de fórmulas de pintura más completa...")
    new_df = generate_complex_paint_formulas(num_samples=15000)

    output_path = os.path.join(output_dir, 'simulated_paint_formulas_complex.csv')
    new_df.to_csv(output_path, index=False)
    print(f"\nNueva base de datos guardada exitosamente en: {output_path}")

    print("\n--- Generación de Datos Completada ---")
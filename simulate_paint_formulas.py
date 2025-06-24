import pandas as pd
import numpy as np

# --- 1. Definición de Parámetros y Rangos (Manteniendo lo anterior) ---

resin_data = {
    'Acrylic': ['SupplierA_Resin', 'SupplierB_Resin'],
    'Alkyd': ['SupplierC_Resin', 'SupplierD_Resin'],
    'Epoxy': ['SupplierE_Resin', 'SupplierF_Resin']
}
pigment_data = {
    'Titanium Dioxide': ['SupplierG_Pigment', 'SupplierH_Pigment'],
    'Iron Oxide Red': ['SupplierI_Pigment', 'SupplierJ_Pigment'],
    'Organic Blue': ['SupplierK_Pigment', 'SupplierL_Pigment']
}
solvent_data = {
    'Water': ['SupplierM_Solvent'],
    'Mineral Spirits': ['SupplierN_Solvent', 'SupplierO_Solvent'],
    'Xylene': ['SupplierP_Solvent', 'SupplierQ_Solvent']
}
additive_types = ['Dispersant', 'Thickener', 'Defoamer']
additive_suppliers = {
    'Dispersant': ['SupplierR_Add', 'SupplierS_Add'],
    'Thickener': ['SupplierT_Add', 'SupplierU_Add'],
    'Defoamer': ['SupplierV_Add', 'SupplierW_Add']
}

min_resin, max_resin = 40, 55
min_pigment, max_pigment = 20, 35
min_solvent, max_solvent = 10, 25
min_additive, max_additive = 0.5, 5

supplier_impact = {
    'SupplierA_Resin': {'gloss_boost': 5, 'visc_boost': 100, 'hiding_boost': 0},
    'SupplierB_Resin': {'gloss_boost': -2, 'visc_boost': -50, 'hiding_boost': 0},
    'SupplierG_Pigment': {'gloss_boost': -5, 'visc_boost': 200, 'hiding_boost': 10},
    'SupplierH_Pigment': {'gloss_boost': 2, 'visc_boost': -80, 'hiding_boost': 5},
    'SupplierN_Solvent': {'gloss_boost': 3, 'visc_boost': -150, 'hiding_boost': 0},
    'SupplierO_Solvent': {'gloss_boost': -1, 'visc_boost': 50, 'hiding_boost': 0},
    'SupplierR_Add': {'visc_boost': -20, 'hiding_boost': 0},
    'SupplierT_Add': {'visc_boost': 300, 'hiding_boost': 0},
    'SupplierV_Add': {'visc_boost': -10, 'hiding_boost': 0},
}

# --- 2. Función para Generar una Sola Fórmula (Modificada) ---

# Probabilidad de que un dato sea faltante (ajusta según desees más o menos faltantes)
MISSING_DATA_PROB = 0.05 # 5% de probabilidad de que una celda sea NaN

# Probabilidad base de éxito (para introducir desbalanceo)
# Ajusta para crear un desbalance. Aquí, 20% éxito, 80% falla.
SUCCESS_BASE_PROB = 0.20 

def generate_single_formula_with_challenges(resin_data, pigment_data, solvent_data, additive_types, additive_suppliers):
    # Seleccionar tipos y proveedores aleatorios
    resin_type = np.random.choice(list(resin_data.keys()))
    resin_supplier = np.random.choice(resin_data[resin_type])

    pigment_type = np.random.choice(list(pigment_data.keys()))
    pigment_supplier = np.random.choice(pigment_data[pigment_type])

    solvent_type = np.random.choice(list(solvent_data.keys()))
    solvent_supplier = np.random.choice(solvent_data[solvent_type])

    chosen_additives = np.random.choice(additive_types, size=np.random.randint(1, 3), replace=False).tolist()
    additive_details = {}
    for add_type in additive_types:
        additive_details[f'{add_type}_Percentage'] = 0.0
        additive_details[f'{add_type}_Supplier'] = 'None'

    additive_total_perc = 0
    for add_type in chosen_additives:
        add_perc = np.random.uniform(min_additive, max_additive)
        add_supplier = np.random.choice(additive_suppliers[add_type])
        additive_details[f'{add_type}_Percentage'] = round(add_perc, 2)
        additive_details[f'{add_type}_Supplier'] = add_supplier
        additive_total_perc += add_perc

    p_resin = np.random.uniform(min_resin, max_resin)
    p_pigment = np.random.uniform(min_pigment, max_pigment)
    p_solvent = np.random.uniform(min_solvent, max_solvent)

    remaining_for_main = 100.0 - additive_total_perc
    if remaining_for_main <= 0:
        remaining_for_main = 1.0

    total_main_perc = p_resin + p_pigment + p_solvent
    
    if total_main_perc == 0:
        p_resin = remaining_for_main / 3
        p_pigment = remaining_for_main / 3
        p_solvent = remaining_for_main / 3
    else:
        p_resin = (p_resin / total_main_perc) * remaining_for_main
        p_pigment = (p_pigment / total_main_perc) * remaining_for_main
        p_solvent = (p_solvent / total_main_perc) * remaining_for_main

    percentages = [p_resin, p_pigment, p_solvent]
    percentages = [round(p, 2) for p in percentages]
    
    current_total = sum(percentages) + additive_total_perc
    difference = 100.0 - current_total
    
    if percentages:
        max_perc_idx = np.argmax(percentages)
        percentages[max_perc_idx] = round(percentages[max_perc_idx] + difference, 2)
    
    p_resin, p_pigment, p_solvent = percentages

    # --- SIMULACIÓN DE PROPIEDADES ---
    gloss = 0
    viscosity = 0
    hiding_power = 0

    # Impacto de la resina
    if resin_type == 'Epoxy':
        gloss += np.random.uniform(70, 95) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(20, 30)
    elif resin_type == 'Acrylic':
        gloss += np.random.uniform(50, 80) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(15, 25)
    else: # Alkyd
        gloss += np.random.uniform(30, 60) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(10, 20)
    
    # Impacto del pigmento
    if pigment_type == 'Titanium Dioxide':
        gloss -= np.random.uniform(0, 10) * (p_pigment / 40)
        viscosity += p_pigment * np.random.uniform(15, 25)
        hiding_power += np.random.uniform(80, 100) * (p_pigment / 40)
    elif pigment_type == 'Organic Blue':
        gloss += np.random.uniform(0, 5) * (p_pigment / 40)
        viscosity += p_pigment * np.random.uniform(10, 20)
        hiding_power += np.random.uniform(40, 60) * (p_pigment / 40)
    else: # Iron Oxide Red
        gloss -= np.random.uniform(0, 3) * (p_pigment / 40)
        viscosity += p_pigment * np.random.uniform(12, 22)
        hiding_power += np.random.uniform(60, 80) * (p_pigment / 40)
        
    # Impacto del solvente
    viscosity -= p_solvent * np.random.uniform(25, 35)
    
    # Añadir impacto de proveedores
    if resin_supplier in supplier_impact:
        gloss += supplier_impact[resin_supplier].get('gloss_boost', 0)
        viscosity += supplier_impact[resin_supplier].get('visc_boost', 0)
        hiding_power += supplier_impact[resin_supplier].get('hiding_boost', 0)
    if pigment_supplier in supplier_impact:
        gloss += supplier_impact[pigment_supplier].get('gloss_boost', 0)
        viscosity += supplier_impact[pigment_supplier].get('visc_boost', 0)
        hiding_power += supplier_impact[pigment_supplier].get('hiding_boost', 0)
    if solvent_supplier in supplier_impact:
        gloss += supplier_impact[solvent_supplier].get('gloss_boost', 0)
        viscosity += supplier_impact[solvent_supplier].get('visc_boost', 0)
        hiding_power += supplier_impact[solvent_supplier].get('hiding_boost', 0)

    # Impacto de aditivos
    for add_type in chosen_additives:
        add_perc = additive_details[f'{add_type}_Percentage']
        add_supplier = additive_details[f'{add_type}_Supplier']
        
        if add_type == 'Dispersant':
            viscosity -= add_perc * np.random.uniform(5, 15)
            hiding_power += add_perc * np.random.uniform(1, 3)
        elif add_type == 'Thickener':
            viscosity += add_perc * np.random.uniform(100, 250)
        elif add_type == 'Defoamer':
            viscosity -= add_perc * np.random.uniform(2, 8)
        
        if add_supplier in supplier_impact:
             viscosity += supplier_impact[add_supplier].get('visc_boost', 0) * (add_perc / 5)
             hiding_power += supplier_impact[add_supplier].get('hiding_boost', 0) * (add_perc / 5)

    # Añadir ruido aleatorio final para realismo
    gloss = max(0, min(100, round(gloss + np.random.normal(0, 5), 2)))
    viscosity = max(100, round(viscosity + np.random.normal(0, 150), 2))
    hiding_power = max(0, min(100, round(hiding_power + np.random.normal(0, 5), 2)))

    # --- SIMULACIÓN DE ÉXITO O FALLA (AJUSTADO PARA DESBALANCE) ---
    is_base_success = (gloss >= 70 and gloss <= 95 and
                       viscosity >= 800 and viscosity <= 1500 and
                       hiding_power >= 85)
    
    # Decidir si la fórmula es un "Éxito" o "Falla" basándose en SUCCESS_BASE_PROB
    # y ajustando por la base_success. Esto crea el desbalance.
    if np.random.rand() < SUCCESS_BASE_PROB:
        # Intentamos generar un ÉXITO. Si la base ya es éxito, genial.
        # Si no lo es, forzamos un poco (o introducimos un "éxito inesperado").
        is_success = True
    else:
        # Intentamos generar una FALLA. Si la base ya es falla, genial.
        # Si la base era éxito, forzamos la falla para mantener el desbalance.
        is_success = False

    formula_data = {
        'ResinType': resin_type,
        'ResinPercentage': p_resin,
        'ResinSupplier': resin_supplier,
        'PigmentType': pigment_type,
        'PigmentPercentage': p_pigment,
        'PigmentSupplier': pigment_supplier,
        'SolventType': solvent_type,
        'SolventPercentage': p_solvent,
        'SolventSupplier': solvent_supplier,
        'Gloss': gloss,
        'Viscosity': viscosity,
        'HidingPower': hiding_power,
        'IsSuccess': int(is_success) # Convertir a 0 o 1
    }
    
    # Añadir detalles de aditivos
    for add_type in additive_types:
        formula_data[f'{add_type}_Percentage'] = additive_details[f'{add_type}_Percentage']
        formula_data[f'{add_type}_Supplier'] = additive_details[f'{add_type}_Supplier']

    # --- Introducir Datos Faltantes Aleatoriamente ---
    # Convertimos los datos a un Series para manipularlo fácilmente
    data_series = pd.Series(formula_data)
    
    # Columnas donde queremos introducir NaNs. Evitamos las columnas de tipo/proveedor por ahora.
    # Podrías incluirlas si quieres simular datos de entrada faltantes.
    columns_to_possibly_miss = [
        'ResinPercentage', 'PigmentPercentage', 'SolventPercentage',
        'Gloss', 'Viscosity', 'HidingPower'
    ]
    for add_type in additive_types:
        columns_to_possibly_miss.append(f'{add_type}_Percentage')

    for col in columns_to_possibly_miss:
        if np.random.rand() < MISSING_DATA_PROB:
            data_series[col] = np.nan # Asignar NaN

    return data_series.to_dict()

# --- 3. Generación del Dataset Completo ---

num_formulas = 5000 

data = []
for _ in range(num_formulas):
    formula = generate_single_formula_with_challenges(
        resin_data, pigment_data, solvent_data, additive_types, additive_suppliers
    )
    data.append(formula)

df = pd.DataFrame(data)

print("Primeras 5 filas del dataset con desafíos:")
print(df.head())

print("\nResumen estadístico de las columnas numéricas (observa los 'count' para NaNs):")
print(df.describe())

print("\nConteo de valores faltantes por columna:")
print(df.isnull().sum())

print("\nConteo de Éxito/Falla (desbalanceado):")
print(df['IsSuccess'].value_counts())
print(f"Porcentaje de Éxito: {df['IsSuccess'].value_counts(normalize=True)[1]:.2f}%")
print(f"Porcentaje de Falla: {df['IsSuccess'].value_counts(normalize=True)[0]:.2f}%")


# Verificación de la suma de porcentajes (después de posibles NaNs, esto puede fallar si hay NaNs)
# Para una verificación precisa, primero imputamos temporalmente o filtramos NaNs
df_temp_for_sum = df.copy()
# Reemplazar NaN en porcentajes por 0 para la suma, asumiendo que NaN significa "no se agregó" o es 0.
# Ojo: esto es solo para la verificación, no para el análisis de ML.
for col in ['ResinPercentage', 'PigmentPercentage', 'SolventPercentage'] + [f'{add}_Percentage' for add in additive_types]:
    df_temp_for_sum[col] = df_temp_for_sum[col].fillna(0)

df_temp_for_sum['TotalPercentage'] = (
    df_temp_for_sum['ResinPercentage'] +
    df_temp_for_sum['PigmentPercentage'] +
    df_temp_for_sum['SolventPercentage']
)
for add_type in additive_types:
    df_temp_for_sum['TotalPercentage'] += df_temp_for_sum[f'{add_type}_Percentage']

print("\nVerificación de la suma total de porcentajes (después de imputar NaNs para suma):")
print(df_temp_for_sum['TotalPercentage'].describe())


# Guardar el dataset
df.to_csv('simulated_paint_formulas_challenges.csv', index=False)
print("\nDataset con desafíos guardado como 'simulated_paint_formulas_challenges.csv'")
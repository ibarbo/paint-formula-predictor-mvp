import pandas as pd
import numpy as np

# --- 1. Definición de Parámetros y Rangos ---
# Estos parámetros definen los tipos de materiales y sus proveedores,
# así como los rangos de porcentajes aceptables para la simulación.

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
additive_types = ['Dispersant', 'Thickener', 'Defoamer'] # Tipos de aditivos
additive_suppliers = {
    'Dispersant': ['SupplierR_Add', 'SupplierS_Add'],
    'Thickener': ['SupplierT_Add', 'SupplierU_Add'],
    'Defoamer': ['SupplierV_Add', 'SupplierW_Add']
}

# Rangos de porcentajes de componentes principales (suman hasta lo que queda después de aditivos)
min_resin, max_resin = 40, 55
min_pigment, max_pigment = 20, 35
min_solvent, max_solvent = 10, 25
min_additive, max_additive = 0.5, 5 # Porcentaje para cada aditivo

# Impacto simulado de los proveedores en las propiedades
# Estos valores son arbitrarios y simulan variaciones de calidad
supplier_impact = {
    'SupplierA_Resin': {'gloss_boost': 5, 'visc_boost': 100, 'hiding_boost': 0},
    'SupplierB_Resin': {'gloss_boost': -2, 'visc_boost': -50, 'hiding_boost': 0},
    'SupplierG_Pigment': {'gloss_boost': -5, 'visc_boost': 200, 'hiding_boost': 10},
    'SupplierH_Pigment': {'gloss_boost': 2, 'visc_boost': -80, 'hiding_boost': 5},
    'SupplierN_Solvent': {'gloss_boost': 3, 'visc_boost': -150, 'hiding_boost': 0},
    'SupplierO_Solvent': {'gloss_boost': -1, 'visc_boost': 50, 'hiding_boost': 0},
    'SupplierR_Add': {'visc_boost': -20, 'hiding_boost': 0}, # Dispersant: reduce viscosidad
    'SupplierT_Add': {'visc_boost': 300, 'hiding_boost': 0}, # Thickener: aumenta viscosidad
    'SupplierV_Add': {'visc_boost': -10, 'hiding_boost': 0}, # Defoamer: ligera reducción
}

# Probabilidad de que un dato numérico sea faltante (simulando errores de registro)
MISSING_DATA_PROB = 0.05 # 5% de probabilidad

# Probabilidad base de que una fórmula sea "Éxito" (para crear desbalance de clases)
SUCCESS_BASE_PROB = 0.20 # 20% de éxito, 80% de falla


# --- 2. Función para Generar una Sola Fórmula (Modificada) ---
# Esta función crea una única entrada de datos con componentes, proveedores,
# propiedades calculadas y las nuevas características de ingeniería.
# Además, introduce datos faltantes y desbalance de clases.

def generate_single_formula_with_challenges(resin_data, pigment_data, solvent_data, additive_types, additive_suppliers):
    """
    Genera una única entrada de datos para una fórmula de pintura, incluyendo:
    - Tipos y porcentajes de resina, pigmento y solvente.
    - Proveedores para cada componente principal.
    - Presencia y porcentaje de aditivos con sus proveedores.
    - Propiedades de la pintura (Brillo, Viscosidad, Poder Cubriente).
    - Un indicador binario de Éxito/Falla.
    - Nuevas características de ingeniería (TSC, PVC simulado, Ratio Solvente-Resina, Combinaciones).
    - Introducción aleatoria de valores faltantes.
    - Desbalance en la probabilidad de Éxito/Falla.
    """
    # Selección aleatoria de tipos y proveedores para componentes principales
    resin_type = np.random.choice(list(resin_data.keys()))
    resin_supplier = np.random.choice(resin_data[resin_type])

    pigment_type = np.random.choice(list(pigment_data.keys()))
    pigment_supplier = np.random.choice(pigment_data[pigment_type])

    solvent_type = np.random.choice(list(solvent_data.keys()))
    solvent_supplier = np.random.choice(solvent_data[solvent_type])

    # Gestión de aditivos: Se eligen entre 1 y 2 aditivos por fórmula
    chosen_additives = np.random.choice(additive_types, size=np.random.randint(1, 3), replace=False).tolist()
    additive_details = {} # Diccionario para almacenar porcentaje y proveedor de cada aditivo
    for add_type in additive_types:
        additive_details[f'{add_type}_Percentage'] = 0.0
        additive_details[f'{add_type}_Supplier'] = 'None' # Valor por defecto si el aditivo no se usa

    additive_total_perc = 0 # Suma de los porcentajes de los aditivos seleccionados
    for add_type in chosen_additives:
        add_perc = np.random.uniform(min_additive, max_additive)
        add_supplier = np.random.choice(additive_suppliers[add_type])
        additive_details[f'{add_type}_Percentage'] = round(add_perc, 2)
        additive_details[f'{add_type}_Supplier'] = add_supplier
        additive_total_perc += add_perc

    # Generación inicial de porcentajes para componentes principales
    p_resin = np.random.uniform(min_resin, max_resin)
    p_pigment = np.random.uniform(min_pigment, max_pigment)
    p_solvent = np.random.uniform(min_solvent, max_solvent)

    # Normalización de porcentajes para que la suma de todos los componentes sea 100%
    # Primero, calculamos el porcentaje restante para los componentes principales
    remaining_for_main = 100.0 - additive_total_perc
    if remaining_for_main <= 0: # Caso de borde si los aditivos son demasiados
        remaining_for_main = 1.0

    total_main_perc = p_resin + p_pigment + p_solvent
    
    # Ajuste de los porcentajes principales para que sumen 'remaining_for_main'
    if total_main_perc == 0: # Evitar división por cero
        p_resin = remaining_for_main / 3
        p_pigment = remaining_for_main / 3
        p_solvent = remaining_for_main / 3
    else:
        p_resin = (p_resin / total_main_perc) * remaining_for_main
        p_pigment = (p_pigment / total_main_perc) * remaining_for_main
        p_solvent = (p_solvent / total_main_perc) * remaining_for_main

    # Redondeo y ajuste final para asegurar la suma exacta a 100% (incluyendo aditivos)
    percentages = [p_resin, p_pigment, p_solvent]
    percentages = [round(p, 2) for p in percentages]
    
    current_total = sum(percentages) + additive_total_perc
    difference = 100.0 - current_total
    
    if percentages:
        # Distribuye la pequeña diferencia del redondeo al componente principal más grande
        max_perc_idx = np.argmax(percentages)
        percentages[max_perc_idx] = round(percentages[max_perc_idx] + difference, 2)
    
    p_resin, p_pigment, p_solvent = percentages

    # --- SIMULACIÓN DE PROPIEDADES INICIALES (Brillo, Viscosidad, Poder Cubriente) ---
    # Las propiedades son simuladas basándose en el tipo y porcentaje de componentes,
    # y los factores de impacto de los proveedores.
    gloss = 0
    viscosity = 0
    hiding_power = 0

    # Impacto base de la resina
    if resin_type == 'Epoxy':
        gloss += np.random.uniform(70, 95) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(20, 30)
    elif resin_type == 'Acrylic':
        gloss += np.random.uniform(50, 80) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(15, 25)
    else: # Alkyd
        gloss += np.random.uniform(30, 60) * (p_resin / 100)
        viscosity += p_resin * np.random.uniform(10, 20)
    
    # Impacto base del pigmento
    if pigment_type == 'Titanium Dioxide':
        gloss -= np.random.uniform(0, 10) * (p_pigment / 40) # Mayor % pigmento puede reducir brillo
        viscosity += p_pigment * np.random.uniform(15, 25)
        hiding_power += np.random.uniform(80, 100) * (p_pigment / 40) # Dióxido de titanio es clave para cubriente
    elif pigment_type == 'Organic Blue':
        gloss += np.random.uniform(0, 5) * (p_pigment / 40)
        viscosity += p_pigment * np.random.uniform(10, 20)
        hiding_power += np.random.uniform(40, 60) * (p_pigment / 40)
    else: # Iron Oxide Red
        gloss -= np.random.uniform(0, 3) * (p_pigment / 40)
        viscosity += p_pigment * np.random.uniform(12, 22)
        hiding_power += np.random.uniform(60, 80) * (p_pigment / 40)
        
    # Impacto base del solvente
    viscosity -= p_solvent * np.random.uniform(25, 35) # Los solventes reducen la viscosidad

    # Añadir impacto de proveedores (modela la variación de calidad de materias primas)
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

    # Impacto de aditivos (dependiendo de su tipo y porcentaje)
    for add_type in chosen_additives:
        add_perc = additive_details[f'{add_type}_Percentage']
        add_supplier = additive_details[f'{add_type}_Supplier']
        
        if add_type == 'Dispersant':
            viscosity -= add_perc * np.random.uniform(5, 15) # Dispersantes reducen viscosidad
            hiding_power += add_perc * np.random.uniform(1, 3) # Mejoran la dispersión, impactando cubriente
        elif add_type == 'Thickener':
            viscosity += add_perc * np.random.uniform(100, 250) # Espesantes aumentan mucho la viscosidad
        elif add_type == 'Defoamer':
            viscosity -= add_perc * np.random.uniform(2, 8) # Antiespumantes tienen impacto menor en viscosidad
        
        if add_supplier in supplier_impact:
             viscosity += supplier_impact[add_supplier].get('visc_boost', 0) * (add_perc / 5)
             hiding_power += supplier_impact[add_supplier].get('hiding_boost', 0) * (add_perc / 5)

    # Añadir ruido aleatorio final para simular variabilidad inherente en el proceso
    gloss = max(0, min(100, round(gloss + np.random.normal(0, 5), 2)))
    viscosity = max(100, round(viscosity + np.random.normal(0, 150), 2))
    hiding_power = max(0, min(100, round(hiding_power + np.random.normal(0, 5), 2)))

    # --- SIMULACIÓN DE ÉXITO O FALLA (AJUSTADO PARA DESBALANCE) ---
    # Define si una fórmula cumple los criterios de calidad deseados.
    is_base_success_criteria_met = (gloss >= 70 and gloss <= 95 and
                                    viscosity >= 800 and viscosity <= 1500 and
                                    hiding_power >= 85)
    
    # Aplica la probabilidad base para crear el desbalance de clases.
    # Si np.random.rand() < SUCCESS_BASE_PROB, intentamos que sea un éxito.
    # Si no, intentamos que sea una falla. Esto predomina sobre el criterio base
    # para asegurar el desbalance deseado.
    if np.random.rand() < SUCCESS_BASE_PROB:
        is_success = True
    else:
        is_success = False

    # Datos base de la fórmula
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
        'IsSuccess': int(is_success) # 0 para falla, 1 para éxito
    }
    
    # Añadir detalles de aditivos al diccionario de datos
    for add_type in additive_types:
        formula_data[f'{add_type}_Percentage'] = additive_details[f'{add_type}_Percentage']
        formula_data[f'{add_type}_Supplier'] = additive_details[f'{add_type}_Supplier']

    # --- Feature Engineering: Creación de Nuevas Características ---
    # Estas características se calculan a partir de los datos existentes y
    # capturan insights de formulación. Se asume que los porcentajes usados aquí
    # son los "reales" antes de introducir NaNs para el cálculo de FE.

    # 1. Total Solids Content (TSC): Suma de todos los componentes no volátiles.
    # Se usan los porcentajes calculados antes de la posible introducción de NaNs.
    total_solids_content = (
        p_resin + p_pigment +
        additive_details['Dispersant_Percentage'] +
        additive_details['Thickener_Percentage'] +
        additive_details['Defoamer_Percentage']
    )
    formula_data['TotalSolidsContent'] = round(total_solids_content, 2)

    # 2. Simulated Pigment Volume Concentration (PVC):
    # Proporción del pigmento dentro del total de sólidos. Importante para propiedades de la película.
    # Se maneja la división por cero si no hay sólidos.
    simulated_pvc = (
        p_pigment / total_solids_content
        if total_solids_content > 0 else 0
    )
    formula_data['SimulatedPVC'] = round(simulated_pvc, 4) # Más decimales para precisión

    # 3. Solvent-to-Resin Ratio: Proporción de solvente respecto a la resina.
    # Afecta la viscosidad y aplicación.
    # Se maneja la división por cero si no hay resina.
    solvent_to_resin_ratio = (
        p_solvent / p_resin
        if p_resin > 0 else np.nan # np.nan si no hay resina para evitar inf/div by zero
    )
    formula_data['SolventToResinRatio'] = round(solvent_to_resin_ratio, 2) if not np.isnan(solvent_to_resin_ratio) else np.nan

    # 4. Interacciones Categóricas: Combinaciones de tipos de componentes.
    # Capturan sinergias o incompatibilidades específicas.
    formula_data['Resin_Pigment_Combo'] = f"{resin_type}_{pigment_type}"
    formula_data['Resin_Solvent_Combo'] = f"{resin_type}_{solvent_type}"
    formula_data['Pigment_Solvent_Combo'] = f"{pigment_type}_{solvent_type}" # Agregamos una extra por interés

    # --- Introducir Datos Faltantes Aleatoriamente ---
    # Esta es la última parte, donde algunos valores calculados se convierten a NaN.
    # Esto simula datos incompletos del mundo real.
    data_series = pd.Series(formula_data)
    
    # Columnas donde se introducirán NaNs. Se incluyen las nuevas características numéricas.
    columns_to_possibly_miss = [
        'ResinPercentage', 'PigmentPercentage', 'SolventPercentage',
        'Gloss', 'Viscosity', 'HidingPower',
        'TotalSolidsContent', 'SimulatedPVC', 'SolventToResinRatio'
    ]
    for add_type in additive_types:
        columns_to_possibly_miss.append(f'{add_type}_Percentage')

    for col in columns_to_possibly_miss:
        if np.random.rand() < MISSING_DATA_PROB:
            data_series[col] = np.nan # Asignar NaN

    return data_series.to_dict()

# --- 3. Generación del Dataset Completo ---
# Se define el número de fórmulas a generar y se itera para construir el DataFrame.

num_formulas = 5000 # Número total de entradas de datos a simular

data = [] # Lista para almacenar los diccionarios de fórmulas generadas
for _ in range(num_formulas):
    formula = generate_single_formula_with_challenges(
        resin_data, pigment_data, solvent_data, additive_types, additive_suppliers
    )
    data.append(formula)

df = pd.DataFrame(data) # Crea el DataFrame de Pandas a partir de la lista de diccionarios

# --- Verificaciones y Guardado del Dataset ---
# Se imprimen las primeras filas, estadísticas y conteos de NaNs para una revisión rápida.

print("Primeras 5 filas del dataset avanzado con desafíos y nuevas features:")
print(df.head())

print("\nResumen estadístico de las columnas numéricas (observa los 'count' para NaNs):")
print(df.describe())

print("\nConteo de valores faltantes por columna:")
print(df.isnull().sum())

print("\nPorcentaje de valores faltantes por columna:")
print((df.isnull().sum() / len(df)) * 100)

print("\nConteo de Éxito/Falla (desbalanceado):")
print(df['IsSuccess'].value_counts())
print(f"Porcentaje de Éxito: {df['IsSuccess'].value_counts(normalize=True)[1]:.2f}%")
print(f"Porcentaje de Falla: {df['IsSuccess'].value_counts(normalize=True)[0]:.2f}%")


# Verificación de la suma de porcentajes (después de posibles NaNs, esto puede requerir imputación temporal)
# Para una verificación precisa de que los porcentajes "originales" suman 100,
# se crea una copia y se imputan los NaNs con 0 para el cálculo de la suma.
df_temp_for_sum = df.copy()
percentage_cols_for_sum = ['ResinPercentage', 'PigmentPercentage', 'SolventPercentage'] + \
                          [f'{add}_Percentage' for add in additive_types]
for col in percentage_cols_for_sum:
    df_temp_for_sum[col] = df_temp_for_sum[col].fillna(0) # Tratar NaN como 0 para la suma aquí

df_temp_for_sum['CalculatedTotalPercentage'] = (
    df_temp_for_sum['ResinPercentage'] +
    df_temp_for_sum['PigmentPercentage'] +
    df_temp_for_sum['SolventPercentage']
)
for add_type in additive_types:
    df_temp_for_sum['CalculatedTotalPercentage'] += df_temp_for_sum[f'{add_type}_Percentage']

print("\nVerificación de la suma total de porcentajes (después de imputar NaNs con 0 para este cálculo):")
print(df_temp_for_sum['CalculatedTotalPercentage'].describe())


# Guardar el dataset en un archivo CSV
df.to_csv('simulated_paint_formulas_with_engineered_features.csv', index=False)
print("\nDataset con desafíos y características de ingeniería guardado como 'simulated_paint_formulas_with_engineered_features.csv'")
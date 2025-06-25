import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- Configuración de rutas (AJUSTADA: SOLO SUBE UN NIVEL) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..')) 

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'simple_shap_model')
SIMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_simple_model.joblib')
SIMPLE_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'simple_preprocessor.joblib')


# --- Carga de Modelos y Preprocesadores ---
@st.cache_resource
def load_model_and_preprocessor():
    try:
        st.info(f"DEBUG: Intentando cargar preprocesador desde: {SIMPLE_PREPROCESSOR_PATH}") # DEBUG LINE
        preprocessor = joblib.load(SIMPLE_PREPROCESSOR_PATH)
        st.info(f"DEBUG: Intentando cargar modelo desde: {SIMPLE_MODEL_PATH}") # DEBUG LINE
        model = joblib.load(SIMPLE_MODEL_PATH)
        st.success("Modelos y preprocesador cargados exitosamente.")
        return model, preprocessor
    except FileNotFoundError:
        st.error(f"Error: Modelo o preprocesador no encontrado. Asegúrate de que los archivos existan en: {MODELS_DIR}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo o preprocesador: {e}")
        st.stop()

model, preprocessor = load_model_and_preprocessor()

# --- Definición de Características Esperadas (DEBE COINCIDIR CON train_simple_model.py) ---
numerical_features = [
    'ResinPercentage', 'PigmentPercentage', 'SolventPercentage', 'AdditivePercentage',
    'ApplicationTemp_C', 'Humidity', 'PHLevel', 'Viscosity', 'DryingTime_Hours',
    'Coverage', 'Gloss', 'Biocide_Percentage', 'Coalescent_Percentage',
    'Defoamer_Percentage', 'Dispersant_Percentage', 'EstimatedDensity', # Engineered feature
    'ResinToPigmentRatio', 'ResinToSolventRatio', 'PigmentToSolventRatio', # Engineered features
    'Surfactant_Percentage', 'Thickener_Percentage', 'TotalAdditivesPercentage', # Engineered feature
    'AcrylicOnWood', 'EpoxyOnMetal', 'HighDryingTime', 'LowApplicationTemp', 'TiO2OnConcrete' # Engineered features
]

categorical_features = [
    'SubstrateType', 'ApplicationMethod', 'Biocide_Supplier', 'Coalescent_Supplier',
    'Defoamer_Supplier', 'Dispersant_Supplier', 'HidingPower',
    'PigmentSupplier', 'PigmentType', 'ResinSupplier', 'ResinType',
    'SolventSupplier', 'SolventType', 'Surfactant_Supplier', 'Thickener_Supplier'
]

# Las características que el modelo espera al final, después de la ingeniería, y en el orden exacto.
# Este orden es el que X_train tenía JUSTO ANTES de entrar al preprocessor.transform()
# Es la unión de todas las columnas (originales + ingenierizadas) antes del OneHotEncoding/Scaling.
expected_final_features_order = numerical_features + categorical_features


# --- Función de Ingeniería de Características (DEBE COINCIDIR CON feature_engineering.py) ---
def apply_feature_engineering(df):
    # Crear una copia para evitar SettingWithCopyWarning y asegurar que el DataFrame original no se modifica
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


# --- Título de la Aplicación ---
st.title("🧪 Predictor de Éxito de Fórmulas de Pintura")
st.markdown("Introduce los parámetros de la fórmula de pintura o sube un CSV para predecir si será un éxito.")

# --- Modo de Predicción ---
prediction_mode = st.radio("Selecciona el modo de predicción:", ("Predicción Individual", "Predicción por Lotes (CSV)"))

if prediction_mode == "Predicción Individual":
    st.header("Entrada de Parámetros Individuales")

    # --- Creación de los Inputs del Usuario ---
    # ... (TU CÓDIGO DE INTERFAZ DE USUARIO PARA ENTRADA INDIVIDUAL - No necesita cambios aquí) ...
    with st.expander("Componentes Principales"):
        col1, col2, col3 = st.columns(3)
        res_type = col1.selectbox('Tipo de Resina', ['Acrylic', 'Epoxy', 'Polyurethane', 'Alkyd', 'Latex', 'Silicone'])
        res_supplier = col2.selectbox('Proveedor de Resina', ['SupplierA', 'SupplierB', 'SupplierC', 'SupplierD_Resin', 'SupplierE_Resin'])
        res_perc = col3.number_input('Porcentaje de Resina (%)', min_value=20.0, max_value=60.0, value=40.0, step=0.1)

        col4, col5, col6 = st.columns(3)
        pig_type = col4.selectbox('Tipo de Pigmento', ['TiO2', 'Iron Oxide', 'Carbon Black', 'Phthalo Blue', 'Quinacridone', 'Metallic'])
        pig_supplier = col5.selectbox('Proveedor de Pigmento', ['SupplierF', 'SupplierG_Pigment', 'SupplierH_Pigment', 'SupplierI_Pigment', 'SupplierJ_Pigment', 'SupplierK_Pigment', 'SupplierL_Pigment'])
        pig_perc = col6.number_input('Porcentaje de Pigmento (%)', min_value=5.0, max_value=40.0, value=20.0, step=0.1)
        
        col7, col8, col9 = st.columns(3)
        sol_type = col7.selectbox('Tipo de Solvente', ['Water', 'Mineral Spirits', 'Acetone', 'Xylene', 'MEK', 'Glycol Ether'])
        sol_supplier = col8.selectbox('Proveedor de Solvente', ['SupplierM', 'SupplierN', 'SupplierO_Solvent', 'SupplierP_Solvent'])
        sol_perc = col9.number_input('Porcentaje de Solvente (%)', min_value=10.0, max_value=50.0, value=30.0, step=0.1)

    with st.expander("Aditivos"):
        col1, col2, col3 = st.columns(3)
        disp_perc = col1.number_input('Dispersant_Percentage', min_value=0.01, max_value=5.0, value=0.5, step=0.01)
        disp_supplier = col2.selectbox('Dispersant_Supplier', ['SupplierQ_Add', 'SupplierR_Add', 'SupplierS_Add'])
        thick_perc = col3.number_input('Thickener_Percentage', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
        
        col4, col5, col6 = st.columns(3)
        thick_supplier = col4.selectbox('Thickener_Supplier', ['SupplierT_Add', 'SupplierU_Add'])
        defo_perc = col5.number_input('Defoamer_Percentage', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        defo_supplier = col6.selectbox('Defoamer_Supplier', ['SupplierV_Add', 'SupplierW_Add'])

        col7, col8, col9 = st.columns(3)
        coal_perc = col7.number_input('Coalescent_Percentage', min_value=0.01, max_value=5.0, value=1.0, step=0.01)
        coal_supplier = col8.selectbox('Coalescent_Supplier', ['SupplierX_Add', 'SupplierY_Add'])
        biocide_perc = col9.number_input('Biocide_Percentage', min_value=0.01, max_value=5.0, value=0.05, step=0.01)

        col10, col11, col12 = st.columns(3)
        biocide_supplier = col10.selectbox('Biocide_Supplier', ['SupplierZ_Add', 'SupplierAA_Add'])
        surf_perc = col11.number_input('Surfactant_Percentage', min_value=0.01, max_value=5.0, value=0.2, step=0.01)
        surf_supplier = col12.selectbox('Surfactant_Supplier', ['SupplierBB_Add', 'SupplierCC_Add'])

    with st.expander("Propiedades y Condiciones de Aplicación"):
        col1, col2, col3 = st.columns(3)
        gloss = col1.number_input('Brillo (Gloss)', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        viscosity = col2.number_input('Viscosidad', min_value=100.0, max_value=10000.0, value=2000.0, step=10.0)
        hiding_power = col3.selectbox('Poder Cubriente (HidingPower)', ['Bajo', 'Medio', 'Alto']) # Ahora categórica
        
        col4, col5, col6 = st.columns(3)
        drying_time = col4.number_input('Tiempo de Secado (Horas)', min_value=0.5, max_value=24.0, value=4.0, step=0.1)
        app_temp = col5.number_input('Temperatura de Aplicación (°C)', min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        substrate_type = col6.selectbox('Tipo de Sustrato', ['Wood', 'Metal', 'Plastic', 'Concrete', 'Drywall'])

        # --- AÑADIDAS LAS NUEVAS COLUMNAS (Que antes faltaban) ---
        col7, col8, col9 = st.columns(3)
        additive_perc = col7.number_input('Porcentaje de Aditivo Total (%)', min_value=0.0, max_value=30.0, value=5.0, step=0.1) # Esta es AdditivePercentage
        ph_level = col8.number_input('Nivel de PH', min_value=5.0, max_value=10.0, value=7.5, step=0.1)
        coverage = col9.number_input('Rendimiento (m²/L)', min_value=1.0, max_value=30.0, value=10.0, step=0.1)

        col10, col11 = st.columns(2)
        humidity = col10.number_input('Humedad (%)', min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        app_method = col11.selectbox('Método de Aplicación', ['Brocha', 'Rodillo', 'Pistola', 'Inmersion'])

    if st.button("Predecir Éxito"):
        # Crear DataFrame con una sola fila para la predicción
        input_data = pd.DataFrame([{
            'ResinType': res_type, 'ResinSupplier': res_supplier, 'ResinPercentage': res_perc,
            'PigmentType': pig_type, 'PigmentSupplier': pig_supplier, 'PigmentPercentage': pig_perc,
            'SolventType': sol_type, 'SolventSupplier': sol_supplier, 'SolventPercentage': sol_perc,
            'Dispersant_Percentage': disp_perc, 'Dispersant_Supplier': disp_supplier,
            'Thickener_Percentage': thick_perc, 'Thickener_Supplier': thick_supplier,
            'Defoamer_Percentage': defo_perc, 'Defoamer_Supplier': defo_supplier,
            'Coalescent_Percentage': coal_perc, 'Coalescent_Supplier': coal_supplier,
            'Biocide_Percentage': biocide_perc, 'Biocide_Supplier': biocide_supplier,
            'Surfactant_Percentage': surf_perc, 'Surfactant_Supplier': surf_supplier,
            'Gloss': gloss, 'Viscosity': viscosity, 'HidingPower': hiding_power,
            'DryingTime_Hours': drying_time, 'ApplicationTemp_C': app_temp, 'SubstrateType': substrate_type,
            # Nuevas columnas
            'AdditivePercentage': additive_perc, 'PHLevel': ph_level, 'Coverage': coverage,
            'Humidity': humidity, 'ApplicationMethod': app_method
        }])

        # Aplicar ingeniería de características
        input_data_engineered = apply_feature_engineering(input_data.copy())
        
        # Asegurar que el orden de las columnas sea el esperado por el preprocesador
        # (Es crucial que expected_final_features_order refleje el orden de las columnas de X_train)
        try:
            # Seleccionar y reordenar las columnas ANTES de pasar al preprocesador
            input_data_ordered = input_data_engineered[expected_final_features_order]
        except KeyError as e:
            st.error(f"Error en la aplicación de características o en el orden de las columnas: {e}. "
                     "Asegúrate de que todas las columnas en 'expected_final_features_order' existen después de la ingeniería.")
            st.stop()


        # Preprocesar los datos
        processed_input = preprocessor.transform(input_data_ordered)

        # Realizar la predicción
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0][1] # Probabilidad de éxito

        st.subheader("Resultado de la Predicción:")
        if prediction == 1:
            st.success(f"La fórmula de pintura es likely un **ÉXITO**! (Probabilidad: {prediction_proba:.2f})")
        else:
            st.error(f"La fórmula de pintura es likely una **FALLA**. (Probabilidad de éxito: {prediction_proba:.2f})")

elif prediction_mode == "Predicción por Lotes (CSV)":
    st.header("Subir Archivo CSV para Predicción por Lotes")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados (primeras 5 filas):")
            st.dataframe(batch_df.head())

            st.write(f"Columnas en CSV cargado: {batch_df.columns.tolist()}")
            
            # --- IMPUTACIÓN DE DATOS (Manejando NaNs si el CSV de entrada los tiene) ---
            # Si tu CSV de prueba tiene NaNs, esto los manejará antes de la ingeniería de características.
            # Idealmente, deberías usar un SimpleImputer en tu ColumnTransformer si tus datos de entrenamiento
            # también tenían NaNs, pero para propósitos de la app, una imputación básica sirve.
            # Aquí asumimos que todos los NaNs están en columnas numéricas o categóricas esperadas.
            for col in batch_df.columns:
                if col in numerical_features and batch_df[col].isnull().any():
                    batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce') # Asegura tipo numérico
                    batch_df[col].fillna(batch_df[col].median(), inplace=True)
                elif col in categorical_features and batch_df[col].isnull().any():
                    batch_df[col].fillna(batch_df[col].mode()[0], inplace=True)
                elif batch_df[col].dtype == 'object': # Convertir object a string para categóricas
                    batch_df[col] = batch_df[col].astype(str)

            # AÑADIDO: Validación de columnas *antes* de la ingeniería de características
            # Estas son las columnas que el CSV RAW original DEBERÍA tener.
            # Puedes obtener esta lista del simulate_complex_paint_formulas.py o del primer df.columns
            # antes de cualquier FE.
            expected_raw_csv_cols = [
                'ResinType', 'ResinSupplier', 'ResinPercentage', 'PigmentType', 'PigmentSupplier',
                'PigmentPercentage', 'SolventType', 'SolventSupplier', 'SolventPercentage',
                'Dispersant_Percentage', 'Dispersant_Supplier', 'Thickener_Percentage',
                'Thickener_Supplier', 'Defoamer_Percentage', 'Defoamer_Supplier',
                'Coalescent_Percentage', 'Coalescent_Supplier', 'Biocide_Percentage',
                'Biocide_Supplier', 'Surfactant_Percentage', 'Surfactant_Supplier',
                'Gloss', 'Viscosity', 'HidingPower', 'DryingTime_Hours',
                'ApplicationTemp_C', 'SubstrateType', 'AdditivePercentage',
                'PHLevel', 'Coverage', 'Humidity', 'ApplicationMethod'
            ]
            
            missing_cols_in_csv = set(expected_raw_csv_cols) - set(batch_df.columns)
            if missing_cols_in_csv:
                st.error(f"ERROR: Tu archivo CSV subido NO contiene todas las columnas RAW esperadas. "
                         f"Faltan: {list(missing_cols_in_csv)}. "
                         "Asegúrate de que tu CSV tenga las mismas columnas que el archivo generado por 'simulate_complex_paint_formulas.py'.")
                st.stop()
            
            # Filtrar el DataFrame cargado para solo incluir las columnas RAW esperadas
            # Esto evita que columnas inesperadas (ej. de una FE anterior) causen problemas.
            batch_df_filtered = batch_df[expected_raw_csv_cols]

            st.write("Tipos de datos del DataFrame del CSV DESPUÉS de la conversión explícita:")
            st.dataframe(batch_df_filtered.dtypes.reset_index().rename(columns={'index': 'Columna', 0: 'Tipo de Dato'}))
            st.write("Primeras 5 filas del DataFrame del CSV DESPUÉS de la conversión explícita:")
            st.dataframe(batch_df_filtered.head())


            # Aplicar ingeniería de características a los datos del lote
            batch_df_engineered = apply_feature_engineering(batch_df_filtered.copy())
            st.write(f"DEBUG: Columnas en batch_data DESPUÉS de la ingeniería: {batch_df_engineered.columns.tolist()}")


            # Asegurar que el orden de las columnas sea el esperado por el preprocesador
            st.write("Procesando datos de lote...")
            st.write("Tipos de datos del DataFrame justo antes de fallar en el preprocesador:")
            st.dataframe(batch_df_engineered[expected_final_features_order].dtypes.reset_index().rename(columns={'index': 'Columna', 0: 'Tipo de Dato'}))
            st.write("Primeras 5 filas del DataFrame justo antes de fallar en el preprocesador:")
            st.dataframe(batch_df_engineered[expected_final_features_order].head())


            # Verificar si faltan columnas después de la ingeniería y antes de reordenar
            current_cols_after_fe = set(batch_df_engineered.columns)
            missing_expected_features = set(expected_final_features_order) - current_cols_after_fe
            if missing_expected_features:
                st.error(f"ERROR: Faltan columnas en el DataFrame después de la ingeniería de características y antes de la predicción: {list(missing_expected_features)}")
                st.stop()


            try:
                # Seleccionar y reordenar las columnas ANTES de pasar al preprocesador
                batch_df_ordered = batch_df_engineered[expected_final_features_order]
                st.write(f"DEBUG: Número de columnas en batch_data DESPUÉS de la selección y reordenamiento: {batch_df_ordered.shape[1]}")

            except KeyError as e:
                st.error(f"Error en la aplicación de características o en el orden de las columnas del CSV: {e}. "
                         "Asegúrate de que todas las columnas en 'expected_final_features_order' existen después de la ingeniería para tu CSV.")
                st.stop()


            # Preprocesar los datos del lote
            processed_batch = preprocessor.transform(batch_df_ordered)

            # Realizar predicciones
            batch_predictions = model.predict(processed_batch)
            batch_probabilities = model.predict_proba(processed_batch)[:, 1] # Probabilidad de éxito

            results_df = batch_df.copy() # Mantener las columnas originales para el reporte
            results_df['Predicted_IsSuccess'] = batch_predictions
            results_df['Success_Probability'] = batch_probabilities

            st.subheader("Resultados de la Predicción por Lotes:")
            st.dataframe(results_df)

            csv_output = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados CSV",
                data=csv_output,
                file_name="predicciones_pintura.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {e}")
            st.exception(e) # Muestra el stack trace completo para depuración

st.markdown("---")
st.markdown("Desarrollado para el debug de SHAP.")
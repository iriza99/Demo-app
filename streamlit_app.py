import streamlit as st
import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import shutil
import zipfile

from autogluon.tabular import TabularPredictor

# -------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de FEV1 (beta)",
    layout="wide"
)

css = """
<style>
.card {
  background-color: #F3F8FE;
  padding: 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
}
.card h2 {
  margin-top: 0;
  font-family: "Arial", sans-serif;
  font-weight: 700;
  color: #1E3A5F;
}
.card p {
  color: #555555;
  margin-bottom: 1.5rem;
}
.train-button {
  background-color: #00BFA5;
  color: white;
  font-weight: bold;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 0.3rem;
  cursor: pointer;
  text-decoration: none;
  font-size: 1rem;
}
.train-button:hover {
  background-color: #00a48d;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------------
def remove_previous_xgb_model():
    model_file = "xgboost_model.json"
    if os.path.exists(model_file):
        os.remove(model_file)

def remove_previous_autogluon_model():
    model_dir = "autogluon_model"
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)

def download_xgb_model(model):
    model_file = "xgboost_model.json"
    model.save_model(model_file)
    with open(model_file, "rb") as f:
        model_bytes = f.read()
    st.download_button(
        label="Descargar modelo XGBoost",
        data=model_bytes,
        file_name=model_file,
        mime="application/octet-stream"
    )

def download_autogluon_model(predictor):
    model_dir = "autogluon_model"
    predictor.save(model_dir)
    zip_filename = "autogluon_model.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=model_dir)
                zipf.write(filepath, arcname=arcname)
    with open(zip_filename, "rb") as f:
        zip_bytes = f.read()
    st.download_button(
        label="Descargar modelo AutoGluon",
        data=zip_bytes,
        file_name=zip_filename,
        mime="application/octet-stream"
    )

def load_xgb_model(uploaded_file):
    temp_name = "temp_xgb_model.json"
    with open(temp_name, "wb") as f:
        f.write(uploaded_file.getvalue())
    bst = xgb.Booster()
    bst.load_model(temp_name)
    return bst

def load_autogluon_model(uploaded_file):
    temp_zip = "temp_autogluon_model.zip"
    with open(temp_zip, "wb") as f:
        f.write(uploaded_file.getvalue())
    extract_dir = "autogluon_model"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    predictor = TabularPredictor.load(extract_dir)
    return predictor

# -------------------------------------------------------------------
# TÍTULO DE LA APP
# -------------------------------------------------------------------
st.title("Predicción de FEV1 (%)")
st.markdown(
    """
    Esta aplicación entrena o carga un modelo para predecir el **FEV1** (Volumen Espiratorio Forzado en el primer segundo).  
    Sube tu archivo CSV con los datos clínicos, selecciona el algoritmo, y realiza predicciones en la pestaña correspondiente.
    """
)

# -------------------------------------------------------------------
# CREACIÓN DE TABS PRINCIPALES
# -------------------------------------------------------------------
tab_entrenar, tab_predecir = st.tabs(["Entrenamiento", "Predicción"])

# -----------------------------------------------------------
# TAB DE ENTRENAMIENTO
# -----------------------------------------------------------
with tab_entrenar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Entrenamiento / Carga de Modelo</h2>", unsafe_allow_html=True)

    # Selección inicial: cargar modelo o entrenar nuevo
    user_choice = st.radio(
        "¿Deseas cargar un modelo existente o entrenar uno nuevo?",
        ("Cargar Modelo", "Entrenar Modelo Nuevo"),
        index=1
    )

    # ----------------------------
    # OPCIÓN 1: Cargar Modelo Existente
    # ----------------------------
    if user_choice == "Cargar Modelo":
        with st.expander("Cargar modelo entrenado", expanded=True):
            model_type_to_load = st.selectbox(
                "Tipo de modelo a cargar:",
                ["XGBoost", "AutoGluon"],
                help="Selecciona el tipo de modelo que deseas cargar."
            )
            uploaded_model_file = st.file_uploader(
                "Selecciona el archivo del modelo (XGBoost: .json | AutoGluon: .zip)",
                type=["json", "zip"],
                key="model_upload"
            )
            if uploaded_model_file is not None:
                if st.button("Cargar Modelo"):
                    with st.spinner("Cargando modelo..."):
                        remove_previous_xgb_model()
                        remove_previous_autogluon_model()

                        if model_type_to_load == "XGBoost":
                            bst = load_xgb_model(uploaded_model_file)
                            st.session_state["modelo"] = bst
                            st.session_state["algo"] = "XGBoost"
                            st.success("Modelo XGBoost cargado correctamente.")
                        else:
                            predictor = load_autogluon_model(uploaded_model_file)
                            st.session_state["modelo"] = predictor
                            st.session_state["algo"] = "AutoGluon"
                            st.success("Modelo AutoGluon cargado correctamente.")

                        # Marcamos como "entrenado" para habilitar la predicción,
                        # pero indicamos que se ha cargado un modelo (no entrenado aquí).
                        st.session_state["trained"] = True
                        st.session_state["model_loaded"] = True

                        # Borramos posibles métricas y X_train (ya que no queremos mostrar nada de eso).
                        st.session_state.pop("xgb_metrics", None)
                        st.session_state.pop("X_train", None)

    # ----------------------------
    # OPCIÓN 2: Entrenar Modelo Nuevo
    # ----------------------------
    else:
        # Cuando entrenamos un modelo, indicamos que no es un modelo cargado.
        st.session_state["model_loaded"] = False

        with st.expander("Entrenar un modelo nuevo", expanded=True):
            uploaded_file = st.file_uploader(
                "Selecciona tu CSV para entrenamiento",
                type=["csv"],
                key="csv_training",
                help="El archivo debe estar separado por ';'."
            )
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, sep=';')
                st.write("Vista previa de los datos:")
                st.dataframe(df.head())
                st.write(f"Número de muestras: {df.shape[0]}")
                st.write(f"Número de columnas: {df.shape[1]}")

                # Guardamos aparte las columnas de interés (Train) si existen
                df_train_info = None
                if all(col in df.columns for col in ["Registro", "Fecha", "FEV1por_Actualpor"]):
                    df_train_info = df[["Registro", "Fecha", "FEV1por_Actualpor"]].copy()

                # Contar pacientes si existe 'Registro'
                if "Registro" in df.columns:
                    num_pacientes = df["Registro"].nunique()
                    st.write(f"Número de pacientes diferentes: {num_pacientes}")

                # Para entrenar el modelo, quitamos 'Fecha' y 'Registro'
                cols_excluir = [col for col in ["Fecha", "Registro"] if col in df.columns]
                if cols_excluir:
                    df = df.drop(columns=cols_excluir)

                # Selectores
                col1, col2 = st.columns(2)
                with col1:
                    target_col = st.selectbox(
                        "Variable objetivo",
                        ["FEV1por_Actualpor"],
                        help="Selecciona la columna que deseas predecir."
                    )
                with col2:
                    algo = st.selectbox(
                        "Algoritmo a entrenar",
                        ["XGBoost", "AutoGluon"],
                        help="XGBoost: rápido y con explicaciones visuales (SHAP).&#10; AutoGluon: mayor precisión."
                    )

                time_limit_minutes = 0
                if algo == "AutoGluon":
                    time_limit_minutes = st.number_input(
                        "Tiempo de entrenamiento (minutos)",
                        min_value=1,
                        max_value=120,
                        value=10,
                        help="Define el tiempo máximo de entrenamiento para AutoGluon."
                    )

                if st.button("Iniciar Entrenamiento", key="train_button"):
                    # Borramos modelo anterior
                    remove_previous_xgb_model()
                    remove_previous_autogluon_model()

                    X = df.drop(columns=[target_col])
                    y = df[target_col]

                    if algo == "XGBoost":
                        with st.spinner("Entrenando XGBoost con validación cruzada (k=10)..."):
                            dtrain = xgb.DMatrix(X, label=y)
                            params = {
                                "objective": "reg:squarederror",
                                "max_depth": 5,
                                "eta": 0.09,
                                "subsample": 0.9,
                                "colsample_bytree": 0.80,
                                "seed": 42,
                                "eval_metric": "rmse"
                            }
                            cv_results = xgb.cv(
                                params=params,
                                dtrain=dtrain,
                                num_boost_round=100,
                                nfold=10,
                                metrics=["rmse"],
                                seed=42
                            )
                            final_train_rmse = cv_results["train-rmse-mean"].iloc[-1]
                            final_test_rmse = cv_results["test-rmse-mean"].iloc[-1]
                            final_model = xgb.train(params, dtrain, num_boost_round=100)

                        st.session_state["modelo"] = final_model
                        st.session_state["algo"] = "XGBoost"
                        st.session_state["X_train"] = X  # Para SHAP
                        st.session_state["trained"] = True
                        st.session_state["model_loaded"] = False

                  



# -----------------------------------------------------------
# TAB DE PREDICCIÓN
# -----------------------------------------------------------
with tab_predecir:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Predicción</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>
        Sube un archivo CSV para generar predicciones con el modelo entrenado.  
        El CSV debe contener las mismas columnas de entrada (excepto la variable objetivo) que se usaron en el entrenamiento.
        </p>
        """,
        unsafe_allow_html=True
    )

    # 1) Subir CSV de test
    pred_file = st.file_uploader("Selecciona tu CSV para predecir", type=["csv"], key="pred")

    # Solo si se sube un archivo
    if pred_file is not None:
        # Cargamos el CSV si es nuevo o cambió el nombre
        if ("df_pred_full" not in st.session_state) or (st.session_state.get("pred_file_name") != pred_file.name):
            df_pred_full = pd.read_csv(pred_file, sep=';')
            st.session_state["df_pred_full"] = df_pred_full
            st.session_state["pred_file_name"] = pred_file.name
            st.session_state["predictions_done"] = False

        # Mensaje informativo (no mostramos tabla)
        st.info("Archivo CSV cargado correctamente. Presiona 'Predecir' para generar predicciones.")

        # 2) Botón "Predecir"
        if st.button("Predecir"):
            if "modelo" not in st.session_state:
                st.warning("No hay un modelo entrenado. Ve a la pestaña de Entrenamiento.")
            else:
                algo_actual = st.session_state["algo"]
                # Se crea un DataFrame sin las columnas 'Fecha' y 'Registro' para la predicción
                df_pred_limpio = st.session_state["df_pred_full"].drop(columns=["Fecha", "Registro"], errors="ignore")

                with st.spinner("Generando predicciones..."):
                    if algo_actual == "XGBoost":
                        dtest = xgb.DMatrix(df_pred_limpio)
                        preds = st.session_state["modelo"].predict(dtest)
                    else:
                        predictor = st.session_state["modelo"]
                        preds = predictor.predict(df_pred_limpio)

                    # Insertar o reemplazar la columna "predicciones"
                    if "predicciones" in st.session_state["df_pred_full"].columns:
                        st.session_state["df_pred_full"].drop(columns=["predicciones"], inplace=True)
                    st.session_state["df_pred_full"].insert(0, "predicciones", preds)

                # Marcamos que ya se han hecho predicciones
                st.session_state["predictions_done"] = True

    else:
        st.info("Por favor, sube un archivo CSV para predecir.")

    # ----------------------------------------------------------------
    # Mostrar la tabla con predicciones y los sub-tabs si ya se generaron
    # ----------------------------------------------------------------
    if (
        "df_pred_full" in st.session_state 
        and st.session_state.get("predictions_done", False) 
        and "predicciones" in st.session_state["df_pred_full"].columns
    ):
        # Tabla con predicciones
        st.markdown("### Resultados con predicciones:")
        st.dataframe(st.session_state["df_pred_full"])
        csv_data = st.session_state["df_pred_full"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar CSV con predicciones",
            data=csv_data,
            file_name="predicciones.csv",
            mime="text/csv"
        )

        # Mostrar los sub-tabs (SHAP y Evolución) solo si existe un modelo entrenado
        if "modelo" in st.session_state:
            algo_actual = st.session_state["algo"]
            if algo_actual == "XGBoost":
                subtab_shap_pred, subtab_evol = st.tabs(["Explicación con SHAP", "Evolución de Pacientes"])
            else:
                (subtab_evol,) = st.tabs(["Evolución de Pacientes"])

            # -----------------------------------------------------------
            # SUB-TAB SHAP (solo para XGBoost)
            # -----------------------------------------------------------
            if algo_actual == "XGBoost":
                with subtab_shap_pred:
                    st.markdown("<h3>Explicación con SHAP</h3>", unsafe_allow_html=True)
                    df_pred_full = st.session_state["df_pred_full"]
                    # Verificar existencia de las columnas necesarias
                    if "Registro" not in df_pred_full.columns or "Fecha" not in df_pred_full.columns:
                        st.warning("No se encontró la columna 'Registro' o 'Fecha' en el CSV de test.")
                    else:
                        # Filtrado por paciente y fecha
                        registros_unicos = df_pred_full["Registro"].astype(str).str.strip().unique()
                        selected_registro = st.selectbox("Selecciona ID Paciente (Registro):", registros_unicos)
                        subdf = df_pred_full[df_pred_full["Registro"].astype(str).str.strip() == selected_registro]
                        fechas_unicas = subdf["Fecha"].unique()
                        selected_fecha = st.selectbox("Selecciona la Fecha:", fechas_unicas)
                        subdf = subdf[subdf["Fecha"] == selected_fecha]
                        if subdf.empty:
                            st.warning("No se encontró esa combinación de registro y fecha.")
                        else:
                            # Reseteamos el índice para que la fila a explicar sea la posición 0
                            subdf_reset = subdf.reset_index(drop=True)
                            # Extraer las características removiendo 'Fecha', 'Registro' y 'predicciones'
                            subdf_features = subdf_reset.drop(columns=["Fecha", "Registro", "predicciones"], errors="ignore")
                            if st.button("Explicar con SHAP", key="shap_button"):
                                with st.spinner("Calculando valores SHAP..."):
                                    explainer = shap.TreeExplainer(st.session_state["modelo"])
                                    shap_values = explainer.shap_values(subdf_features)
                                    st.write(f"Explicación SHAP para paciente {selected_registro}, fecha {selected_fecha}:")
                                    plt.figure(figsize=(6, 4))
                                    shap.force_plot(
                                        explainer.expected_value,
                                        shap_values[0],
                                        subdf_features.iloc[0, :],
                                        matplotlib=True
                                    )
                                    st.pyplot(plt.gcf())
                                    st.write("Desglose de la predicción")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    shap.waterfall_plot(shap.Explanation(
                                        values=shap_values[0],
                                        base_values=explainer.expected_value,
                                        data=subdf_features.iloc[0, :]
                                    ))
                                    st.pyplot(fig)

            # -----------------------------------------------------------
            # SUB-TAB EVOLUCIÓN (Train + Test o solo Test si no hay datos de train)
            # -----------------------------------------------------------
            with subtab_evol:
                st.markdown("<h3>Gráfico de evolución de pacientes</h3>", unsafe_allow_html=True)
                df_pred_full = st.session_state["df_pred_full"]
                df_train_info = st.session_state.get("df_train_info", None)

                if df_train_info is None:
                    st.info("No hay datos anteriores de entrenamiento. Se muestran únicamente las predicciones.")
                    df_evol = df_pred_full.copy()
                    df_evol["Split"] = "Test"
                    # Convertir la columna Fecha si existe
                    if "Fecha" in df_evol.columns:
                        df_evol["Fecha"] = pd.to_datetime(df_evol["Fecha"], format="%d/%m/%Y", errors="coerce")
                    # Si se encuentra la columna 'Registro', se puede filtrar por paciente
                    if "Registro" in df_evol.columns:
                        df_evol["Registro"] = df_evol["Registro"].astype(str).str.strip()
                        selected_registro = st.selectbox("Selecciona el ID del paciente (Registro):", df_evol["Registro"].unique())
                        df_evol = df_evol[df_evol["Registro"] == selected_registro].copy()
                    else:
                        selected_registro = None
                    if not df_evol.empty:
                        plt.figure(figsize=(10, 5))
                        plt.plot(
                            df_evol["Fecha"],
                            df_evol["predicciones"],
                            label="Predicciones",
                            color="red",
                            marker='o'
                        )
                        plt.xlabel("Fecha")
                        plt.ylabel("FEV1")
                        title = f"Evolución de predicción para el paciente {selected_registro}" if selected_registro else "Evolución de predicción"
                        plt.title(title)
                        plt.legend()
                        st.pyplot(plt.gcf())
                    else:
                        st.warning("No se encontraron datos para graficar.")
                else:
                    # Si existen datos de entrenamiento, se combinan con los de test
                    df_train = df_train_info.copy()
                    df_train["predicciones"] = float("nan")
                    df_train["Split"] = "Train"
                    df_train["Fecha"] = pd.to_datetime(df_train["Fecha"], format="%Y-%m-%d", errors="coerce")

                    df_test = df_pred_full.copy()
                    df_test["Split"] = "Test"
                    if "FEV1por_Actualpor" not in df_test.columns:
                        df_test["FEV1por_Actualpor"] = float("nan")
                    df_test["Fecha"] = pd.to_datetime(df_test["Fecha"], format="%d/%m/%Y", errors="coerce")

                    df_evol = pd.concat([df_train, df_test], ignore_index=True)
                    df_evol["Registro"] = df_evol["Registro"].astype(str).str.strip()
                    selected_registro = st.selectbox("Selecciona el ID del paciente (Registro):", df_evol["Registro"].unique())
                    df_filtered = df_evol[df_evol["Registro"] == selected_registro].copy()
                    df_filtered["Fecha"] = pd.to_datetime(df_filtered["Fecha"], errors="coerce")
                    df_filtered.sort_values(["Fecha", "Split"], inplace=True)
                    train_data = df_filtered[df_filtered["Split"] == "Train"]
                    test_data = df_filtered[df_filtered["Split"] == "Test"]

                    if not train_data.empty or not test_data.empty:
                        plt.figure(figsize=(10, 5))
                        if not train_data.empty:
                            plt.plot(
                                train_data["Fecha"],
                                train_data["FEV1por_Actualpor"],
                                label="Valor Real (Train)",
                                color="blue",
                                marker='o'
                            )
                        if not test_data.empty:
                            plt.plot(
                                test_data["Fecha"],
                                test_data["predicciones"],
                                label="Predicciones (Test)",
                                color="red",
                                marker='o'
                            )
                        plt.xlabel("Fecha")
                        plt.ylabel("FEV1")
                        plt.title(f"Evolución del paciente {selected_registro}")
                        plt.legend()
                        st.pyplot(plt.gcf())
                    else:
                        st.warning("No se encontraron datos para graficar para el paciente seleccionado.")

    st.markdown("</div>", unsafe_allow_html=True)













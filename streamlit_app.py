import streamlit as st
import shap
import pandas as pd
import xgboost as xgb

import matplotlib.pyplot as plt

# Configuración general de la página
st.set_page_config(
    page_title="Predicción de FEV1 (beta)",
    layout="wide"
)

# CSS para estilo de “card” y botón personalizado
css = """
<style>
.card {
  background-color: #F3F8FE; /* fondo claro */
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
.selectors {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}
.selector-box {
  flex: 1;
}
label {
  font-weight: 600;
  color: #333;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Título principal y descripción
st.title("Predicción de FEV1 (%)")
st.markdown("""
Esta aplicación entrena un modelo **XGBoost** para predecir el **FEV1** (Volumen Espiratorio Forzado en el primer segundo), 
un indicador clave en el seguimiento de pacientes con trasplante de pulmón.
Sube tu archivo CSV con los datos clínicos, entrena el modelo y luego genera predicciones en la pestaña correspondiente.
""")

# Creamos dos pestañas: “Entrenamiento” y “Predicción”
tab_entrenar, tab_predecir = st.tabs(["Entrenamiento", "Predicción"])

# -----------------------------------------------------------
# TAB DE ENTRENAMIENTO
# -----------------------------------------------------------

st.markdown("<h2>Entrenamiento de Modelo</h2>", unsafe_allow_html=True)
st.markdown("""
<p>
Carga tu dataset y entrena un modelo XGBoost. 
Asegúrate de incluir la columna <strong>FEV1por_Actualpor</strong> como variable objetivo.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Selecciona tu CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # Información sobre los datos
    st.write(f"Número de muestras: {df.shape[0]}")
    st.write(f"Número de columnas: {df.shape[1]}")

    # Verificamos si hay columna "Registro"
    if "Registro" in df.columns:
        num_pacientes = df["Registro"].nunique()
        st.write(f"Número de pacientes diferentes: {num_pacientes}")

    # Excluir columnas no relevantes
    cols_excluir = [col for col in ["Fecha", "Registro"] if col in df.columns]
    if cols_excluir:
        df = df.drop(columns=cols_excluir)

    # Selección de variables
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Variable objetivo", ["FEV1por_Actualpor"])
    with col2:
        algo = st.selectbox("Algoritmo", ["XGBoost"])

    # -------------------------------------------------------
    # ENTRENAMIENTO DEL MODELO
    # -------------------------------------------------------
    if st.button("Iniciar Entrenamiento"):
        with st.spinner("Entrenando el modelo con 900 iteraciones..."):
            X = df.drop(columns=[target_col])
            y = df[target_col]

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

            final_model = xgb.train(params, dtrain, num_boost_round=900)

            # Guardamos el modelo en la sesión
            st.session_state["modelo"] = final_model
            st.session_state["X_train"] = X
            st.session_state["trained"] = True
            st.session_state["show_shap"] = False

        st.success("Modelo entrenado correctamente")

# -------------------------------------------------------
# MOSTRAR IMPORTANCIA DE CARACTERÍSTICAS CON SHAP
# -------------------------------------------------------
if st.session_state.get("trained", False):
    if st.button("Ver Importancia de Características (SHAP)"):
        st.session_state["show_shap"] = True

if st.session_state.get("show_shap", False):
    with st.spinner("Calculando valores SHAP..."):
        explainer = shap.TreeExplainer(st.session_state["modelo"])
        shap_values = explainer.shap_values(st.session_state["X_train"])

        st.write("Gráfico SHAP de la importancia de características:")
        plt.figure(figsize=(4, 3))
        shap.summary_plot(shap_values, st.session_state["X_train"], show=False)
        fig = plt.gcf()
        st.pyplot(fig)

# -----------------------------------------------------------
# TAB DE PREDICCIÓN
# -----------------------------------------------------------
import shap
import matplotlib.pyplot as plt

with tab_predecir:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Predicción</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>
    Sube un archivo CSV para generar predicciones con el modelo entrenado. 
    Debe contener las mismas columnas de entrada (excepto FEV1) que usaste en el entrenamiento.
    </p>
    """, unsafe_allow_html=True)

    # Subir archivo para predecir
    pred_file = st.file_uploader("Selecciona tu CSV para predecir", type=["csv"], key="pred")

    if pred_file is not None:
        df_pred = pd.read_csv(pred_file, sep=';')

        # Eliminamos las columnas "Fecha" y "Registro" si existen
        cols_excluir = [col for col in ["Fecha", "Registro"] if col in df_pred.columns]
        df_pred_limpio = df_pred.drop(columns=cols_excluir, errors="ignore")

        st.write("Vista previa de los datos a predecir:")
        st.dataframe(df_pred.head())

        # Botón para generar predicciones
        if st.button("Predecir"):
            with st.spinner("Generando predicciones..."):
                if "modelo" in st.session_state:
                    # Convertimos los datos a DMatrix
                    dtest = xgb.DMatrix(df_pred_limpio)
                    # Generamos las predicciones usando el modelo entrenado
                    preds = st.session_state["modelo"].predict(dtest)

                    # Insertamos la columna "Predicción" en la primera posición
                    df_pred.insert(0, "Predicción", preds)
                    st.write("Resultados:")
                    st.dataframe(df_pred)

                    # Opción para descargar el CSV con predicciones
                    csv_data = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Descargar CSV con predicciones",
                        data=csv_data,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )

                    # Guardamos los datos en session_state para SHAP
                    st.session_state["df_pred"] = df_pred_limpio
                    st.session_state["predicciones"] = preds
                else:
                    st.warning("No hay un modelo entrenado. Ve a la pestaña de Entrenamiento primero.")
    
    else:
        st.info("Por favor, sube un archivo CSV para predecir.")

    # ----------------------------------------------------------------
    # SELECCIÓN DE UNA FILA PARA SHAP
    # ----------------------------------------------------------------
    if "df_pred" in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>Explicación con SHAP</h3>", unsafe_allow_html=True)

        # Selección de una muestra para explicar
        idx = st.selectbox("Selecciona un índice para analizar con SHAP:", df_pred.index)

        if st.button("Explicar con SHAP"):
            with st.spinner("Calculando valores SHAP..."):
                explainer = shap.TreeExplainer(st.session_state["modelo"])
                shap_values = explainer.shap_values(st.session_state["df_pred"])

                # 🔹 **Gráfico 1: Waterfall SHAP (impacto por variable)**
                st.write(f"Explicación SHAP para la muestra en el índice {idx}:")
                plt.figure(figsize=(6, 4))  
                shap.force_plot(explainer.expected_value, shap_values[idx], st.session_state["df_pred"].iloc[idx, :], matplotlib=True)
                st.pyplot(plt)

                # 🔹 **Gráfico 2: SHAP Summary Plot (scatter)**
                st.write(f"Desglose de la predicción")
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=st.session_state["df_pred"].iloc[idx, :]))
                st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)  # fin de la card



import streamlit as st
import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Importamos AutoGluon
from autogluon.tabular import TabularPredictor

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
Esta aplicación entrena un modelo **XGBoost** o **AutoGluon** para predecir el **FEV1** (Volumen Espiratorio Forzado en el primer segundo),
un indicador clave en el seguimiento de pacientes con trasplante de pulmón.
Sube tu archivo CSV con los datos clínicos, entrena el modelo y luego genera predicciones en la pestaña correspondiente.
""")

# Creamos dos pestañas: “Entrenamiento” y “Predicción”
tab_entrenar, tab_predecir = st.tabs(["Entrenamiento", "Predicción"])

# -----------------------------------------------------------
# TAB DE ENTRENAMIENTO
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# TAB DE ENTRENAMIENTO
# -----------------------------------------------------------

with tab_entrenar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Entrenamiento de Modelo</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>
    Configura y entrena tu modelo de Machine Learning con tu dataset. Primero, carga tu CSV (separado por ';') y asegúrate
    de que contenga la columna <strong>FEV1por_Actualpor</strong> junto con 
    otras variables que consideres relevantes (edad, tiempo post-trasplante, etc.).
    </p>
    """, unsafe_allow_html=True)

    # Si el CSV se borra, reiniciamos todo el estado
    if "uploaded_csv" in st.session_state and st.session_state["uploaded_csv"] is not None:
        if st.session_state["uploaded_csv"] is None:
            st.session_state.clear()
            st.experimental_rerun()

    # Subir dataset
    uploaded_file = st.file_uploader("Selecciona tu CSV", type=["csv"])
    st.session_state["uploaded_csv"] = uploaded_file  # Guardamos en session_state

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        st.write(f"Número de muestras: {df.shape[0]}")
        st.write(f"Número de columnas: {df.shape[1]}")

        # Si existe la columna 'Registro', contamos pacientes distintos
        if "Registro" in df.columns:
            num_pacientes = df["Registro"].nunique()
            st.write(f"Número de pacientes diferentes: {num_pacientes}")
        else:
            st.warning("No se encontró la columna 'Registro' para calcular el número de pacientes diferentes.")

        # Guardamos columnas 'Fecha' y 'Registro' (si existen) para futuro y las quitamos del DF
        cols_excluir = []
        for col in ["Fecha", "Registro"]:
            if col in df.columns:
                cols_excluir.append(col)

        if cols_excluir:
            st.session_state["excluidas"] = df[cols_excluir].copy()
            df = df.drop(columns=cols_excluir)

        # Selectores en dos columnas
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Variable objetivo", ["FEV1por_Actualpor"])
        with col2:
            algo = st.selectbox("Algoritmo", ["XGBoost", "AutoGluon"])

        # Si el usuario elige AutoGluon, permitimos especificar tiempo de entrenamiento (en minutos)
        time_limit_minutes = 0
        if algo == "AutoGluon":
            time_limit_minutes = st.number_input(
                "Tiempo de entrenamiento (minutos)",
                min_value=1, max_value=120, value=10
            )

        # Botón para iniciar el entrenamiento
        if st.button("Iniciar Entrenamiento"):
            X = df.drop(columns=[target_col])
            y = df[target_col]

            if algo == "XGBoost":
                with st.spinner("Entrenando el modelo con XGBoost y validación cruzada (10-fold)..."):
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

                    # Validación cruzada con 10 folds
                    cv_results = xgb.cv(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=1000,
                        nfold=10,
                        metrics=["rmse"],
                        seed=42
                    )

                    # Guardamos el dataframe de resultados en session_state
                    st.session_state["xgb_cv_results"] = cv_results

                    # Entrenamos el modelo final
                    final_model = xgb.train(params, dtrain, num_boost_round=1000)

                # Guardamos en session_state
                st.session_state["modelo"] = final_model
                st.session_state["X_train"] = X
                st.session_state["trained"] = True
                st.session_state["show_shap"] = False
                st.session_state["algo"] = "XGBoost"
                st.success("Modelo XGBoost entrenado con éxito (con CV de 10-fold).")

            else:
                # Entrenamos con AutoGluon
                with st.spinner("Entrenando el modelo con AutoGluon..."):
                    train_data = pd.concat([X, y], axis=1)

                    # Solo usar algoritmos de árboles
                    hyperparameters = {
                        'GBM': {},   # LightGBM
                        'CAT': {},   # CatBoost
                        'XGB': {},   # XGBoost
                    }

                    # Convertimos a segundos
                    time_limit_seconds = time_limit_minutes * 60

                    predictor = TabularPredictor(
                        label=target_col,
                        eval_metric="root_mean_squared_error"
                    ).fit(
                        train_data=train_data,
                        time_limit=time_limit_seconds,
                        hyperparameters=hyperparameters
                    )

                st.session_state["modelo"] = predictor
                st.session_state["X_train"] = X
                st.session_state["trained"] = True
                st.session_state["show_shap"] = False
                st.session_state["algo"] = "AutoGluon"
                st.success("Modelo AutoGluon entrenado con éxito.")

                # Guardamos el leaderboard para mostrarlo después
                st.session_state["autogluon_leaderboard"] = predictor.leaderboard(silent=True)

    else:
        st.info("Por favor, sube un archivo CSV para entrenar el modelo.")

    # ----------------------------------------------------------------
    # MOSTRAR RESULTADOS (XGBoost CV y AutoGluon leaderboard) FUERA DEL BOTÓN
    # ----------------------------------------------------------------
    # 1) Mostramos resultados de XGBoost si existen
    if st.session_state.get("algo") == "XGBoost" and "xgb_cv_results" in st.session_state:
        st.write("## Resultados XGBoost")
        cv_results = st.session_state["xgb_cv_results"]

        # Métricas finales (última fila)
        final_train_rmse = cv_results["train-rmse-mean"].iloc[-1]
        final_test_rmse = cv_results["test-rmse-mean"].iloc[-1]
        st.write(f"**RMSE final (entrenamiento)**: {final_train_rmse:.4f}")
        st.write(f"**RMSE final (validación)**: {final_test_rmse:.4f}")

    # 2) Mostramos leaderboard de AutoGluon si existe
    if st.session_state.get("algo") == "AutoGluon" and "autogluon_leaderboard" in st.session_state:
        st.write("## Leaderboard de modelos - AutoGluon")
        st.dataframe(st.session_state["autogluon_leaderboard"])

    # ----------------------------------------------------------------
    # SHAP para XGBoost (opcional)
    # ----------------------------------------------------------------
    if st.session_state.get("trained", False) and st.session_state.get("algo") == "XGBoost":
        # Botón para ver la importancia de características (SHAP)
        if st.button("Ver Importancia de Características (SHAP)"):
            st.session_state["show_shap"] = True

        if st.session_state.get("show_shap", False):
            with st.spinner("Calculando valores SHAP..."):
                explainer = shap.TreeExplainer(st.session_state["modelo"])
                shap_values = explainer.shap_values(st.session_state["X_train"])

                st.write("Gráfico SHAP de la importancia de las características:")
                plt.figure(figsize=(3, 2))  
                shap.summary_plot(shap_values, st.session_state["X_train"], show=False)
                fig = plt.gcf()
                st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# TAB DE PREDICCIÓN
# -----------------------------------------------------------

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
                    # Distinguimos entre XGBoost y AutoGluon
                    if st.session_state.get("algo") == "XGBoost":
                        dtest = xgb.DMatrix(df_pred_limpio)
                        preds = st.session_state["modelo"].predict(dtest)
                    else:
                        # AutoGluon
                        preds = st.session_state["modelo"].predict(df_pred_limpio)

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

                    # Guardamos los datos en session_state para SHAP (solo si quisiéramos)
                    st.session_state["df_pred"] = df_pred_limpio
                    st.session_state["predicciones"] = preds
                else:
                    st.warning("No hay un modelo entrenado. Ve a la pestaña de Entrenamiento primero.")
    else:
        st.info("Por favor, sube un archivo CSV para predecir.")

    # ----------------------------------------------------------------
    # SELECCIÓN DE UNA FILA PARA SHAP (SOLO XGBOOST EN ESTE EJEMPLO)
    # ----------------------------------------------------------------
    if "df_pred" in st.session_state and st.session_state.get("algo") == "XGBoost":
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>Explicación con SHAP</h3>", unsafe_allow_html=True)

        idx = st.selectbox("Selecciona un índice para analizar con SHAP:", st.session_state["df_pred"].index)

        if st.button("Explicar con SHAP"):
            with st.spinner("Calculando valores SHAP..."):
                explainer = shap.TreeExplainer(st.session_state["modelo"])
                shap_values = explainer.shap_values(st.session_state["df_pred"])

                st.write(f"Explicación SHAP para la muestra en el índice {idx}:")
                plt.figure(figsize=(6, 4))  
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[idx],
                    st.session_state["df_pred"].iloc[idx, :],
                    matplotlib=True
                )
                st.pyplot(plt)

                st.write("Desglose de la predicción")
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values[idx],
                    base_values=explainer.expected_value,
                    data=st.session_state["df_pred"].iloc[idx, :]
                ))
                st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)  # fin de la card

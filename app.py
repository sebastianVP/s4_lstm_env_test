import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CONFIGURACIÓN GLOBAL
# =============================================================================
st.set_page_config(page_title="Predicción S4 Ionosférico", layout="wide")

# Características exactas de entrada a la red LSTM
LSTM_FEATURES = ["S4", "TEC", "daily_sin", "daily_cos", "day_of_year_sin", "day_of_year_cos"]
LOOKBACK = 70
HORIZON = 10
THRESHOLD = 0.6

# =============================================================================
# 2. FUNCIÓN DE PÉRDIDA PERSONALIZADA
# =============================================================================
def focal_mse_s4(threshold_norm, alpha=1.5, beta=25.0, underestimation_penalty=2.0):
    def loss(y_true, y_pred):
        error = y_pred - y_true
        abs_error = tf.abs(error)
        squared_error = tf.square(error)
        
        event_mask = tf.cast(y_true >= threshold_norm, tf.float32)
        event_weights = 1.0 + event_mask * (beta - 1.0)
        focal_weights = tf.pow(abs_error + 1e-7, alpha)
        
        underestimation_mask = tf.cast(tf.logical_and(y_true >= threshold_norm, y_pred < y_true), tf.float32)
        underestimation_weights = 1.0 + underestimation_mask * (underestimation_penalty - 1.0)
        
        total_loss = squared_error * event_weights * focal_weights * underestimation_weights
        return tf.reduce_mean(total_loss)
    return loss

# =============================================================================
# 3. CARGA EN CACHÉ DE RECURSOS (Modelo y Scaler)
# =============================================================================
@st.cache_resource
def load_assets():
    scaler = joblib.load("SCALER_S4.pkl")
    
    # Extraer las columnas exactas con las que se entrenó el Scaler
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
    else:
        scaler_features = LSTM_FEATURES

    # Obtener min y max para desnormalizar S4
    if "S4" in scaler_features:
        idx_s4 = scaler_features.index("S4")
        min_s4 = scaler.data_min_[idx_s4]
        max_s4 = scaler.data_max_[idx_s4]
    else:
        min_s4 = scaler.data_min_[0]
        max_s4 = scaler.data_max_[0]

    threshold_norm = (THRESHOLD - min_s4) / (max_s4 - min_s4)
    
    loss_fn = focal_mse_s4(
        threshold_norm=threshold_norm, 
        alpha=1.5, 
        beta=25.0, 
        underestimation_penalty=2.0
    )
    
    model = load_model(
        "modelo_lstm_experimento_S4_1808.keras", 
        custom_objects={"loss": loss_fn}
    )
    
    # Retorna exactamente 5 elementos
    return scaler, model, min_s4, max_s4, scaler_features

# =============================================================================
# 4. FUNCIONES DE PROCESAMIENTO
# =============================================================================
def add_temporal_features(df):
    """Agrega variables trigonométricas temporales."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)

    minute_of_day = idx.hour * 60 + idx.minute
    daily_period = 24 * 60
    out["daily_sin"] = np.sin(2 * np.pi * minute_of_day / daily_period)
    out["daily_cos"] = np.cos(2 * np.pi * minute_of_day / daily_period)

    day_of_year = idx.dayofyear
    days_in_year = np.where(idx.is_leap_year, 366, 365)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / days_in_year)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / days_in_year)

    return out

def desnormalize_s4(values_scaled, min_s4, max_s4):
    """Convierte los valores de S4 a escala física."""
    return np.abs(values_scaled * (max_s4 - min_s4) + min_s4)

# =============================================================================
# 5. INTERFAZ DE USUARIO (UI)
# =============================================================================
def main():
    st.title("🛰️ Predicción de Centelleo Ionosférico (S4) con LSTM")
    st.markdown("""
    Sube un archivo CSV con las observaciones recientes. El archivo debe contener al menos **70 minutos** 
    ininterrumpidos de datos.
    """)

    # Cargar recursos (Se reciben los 5 elementos correctamente)
    try:
        scaler, model, min_s4, max_s4, scaler_features = load_assets()
        st.sidebar.success("✅ Modelo y Scaler cargados correctamente.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar los archivos: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Selecciona tu archivo de datos (.csv)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Validar existencia de timestamp
        if 'timestamp' not in df.columns and 'Tiempo' not in df.columns:
            st.error("⚠️ El archivo CSV debe contener una columna de fecha/hora ('timestamp' o 'Tiempo').")
            st.stop()

        time_col = 'timestamp' if 'timestamp' in df.columns else 'Tiempo'
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)

        st.subheader("Datos Originales Recientes")
        st.dataframe(df.tail())

        if len(df) < LOOKBACK:
            st.error(f"⚠️ Se necesitan al menos {LOOKBACK} observaciones. El archivo contiene {len(df)}.")
            st.stop()

        if st.button("Generar Predicción a 10 Minutos"):
            with st.spinner("Procesando y calculando..."):
                # 1. Feature Engineering
                df_fe = add_temporal_features(df)
                
                # 2. Verificar que existan en el CSV todas las columnas requeridas por el scaler
                missing_cols = [c for c in scaler_features if c not in df_fe.columns]
                if missing_cols:
                    st.error(f"⚠️ El archivo CSV no contiene las siguientes columnas requeridas por el scaler: {missing_cols}")
                    st.stop()

                # 3. Seleccionar las columnas exactas del scaler para las últimas 70 filas
                window_df = df_fe.iloc[-LOOKBACK:][scaler_features]
                last_timestamp = window_df.index[-1]
                
                # 4. Escalar datos con el conjunto completo de variables
                window_scaled_array = scaler.transform(window_df)
                window_scaled_df = pd.DataFrame(window_scaled_array, columns=scaler_features)
                
                # 5. Filtrar únicamente las 6 columnas de entrada para la red LSTM
                lstm_input_df = window_scaled_df[LSTM_FEATURES]
                
                # 6. Formatear para LSTM: (1, 70, 6)
                X_input = lstm_input_df.to_numpy().reshape(1, LOOKBACK, len(LSTM_FEATURES))
                
                # 7. Predicción y desnormalización
                y_pred_scaled = model.predict(X_input)
                y_pred_real = desnormalize_s4(y_pred_scaled[0], min_s4, max_s4)
                
                # 8. Rango de fechas futuras (10 minutos)
                future_dates = pd.date_range(
                    start=last_timestamp + pd.Timedelta(minutes=1), 
                    periods=HORIZON, 
                    freq='1min'
                )
                
                # 9. CONSTRUCCIÓN DEL DATAFRAME COMBINADO (Pasado + Futuro)
                # Extraemos los 70 minutos pasados reales
                history_s4 = window_df["S4"].copy()
                
                # Crear DataFrame histórico
                df_history = pd.DataFrame({"S4 Histórico (70 min)": history_s4})
                
                # Crear DataFrame futuro
                df_future = pd.DataFrame({"S4 Predicho (10 min)": y_pred_real}, index=future_dates)
                
                # Conectar la línea: agregamos el último valor real al inicio del bloque predicho
                connection_point = pd.DataFrame(
                    {"S4 Predicho (10 min)": [history_s4.iloc[-1]]}, 
                    index=[last_timestamp]
                )
                df_future_connected = pd.concat([connection_point, df_future])
                
                # Unir todo en una sola tabla temporal para el gráfico
                chart_df = pd.concat([df_history, df_future_connected], axis=0)

                # 10. MOSTRAR RESULTADOS
                st.subheader(f"🔮 Ventana Observada y Predicción a partir de {last_timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                # Gráfico interactivo:
                # "#2ECC71" -> Verde (Pasado / Lookback)
                # "#FF4B4B" -> Rojo (Futuro / Predicción)
                st.line_chart(
                    chart_df, 
                    color=["#2ECC71", "#FF4B4B"]
                )
                
                # Tabla de predicción futura únicamente
                pred_df = pd.DataFrame({
                    "Timestamp": future_dates,
                    "Predicción S4": y_pred_real
                }).set_index("Timestamp")
                
                st.subheader("📋 Detalle de la Predicción a 10 Minutos")
                st.dataframe(pred_df.style.highlight_max(axis=0, color="#FFDDDD"))

                # Evaluación del umbral de alerta
                max_pred = pred_df["Predicción S4"].max()
                if max_pred >= THRESHOLD:
                    st.warning(f"⚠️ **ALERTA DE EVENTO:** Se predice un pico de S4 de {max_pred:.4f} que supera el umbral de {THRESHOLD}.")
                else:
                    st.info(f"✅ Condiciones normales. El valor máximo pronosticado es {max_pred:.4f}.")

if __name__ == "__main__":
    main()

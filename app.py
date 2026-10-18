"""
Sistema de Pronóstico de Centelleo Ionosférico (S4) con LSTM
==============================================================

Modo "explorador": subes un tramo histórico (cuantas más filas, mejor --
no solo 360), y dentro de la app te mueves libremente por el tiempo para
ver el pronóstico del modelo superpuesto contra el valor REAL que
efectivamente ocurrió (backtesting visual). Útil para validar el modelo
en distintos regímenes: eventos fuertes vs. períodos tranquilos.

Si subes un archivo con EXACTAMENTE LOOKBACK filas (el caso de producción
real, sin futuro conocido), la app cae automáticamente en modo pronóstico
puro -- sin comparación, porque no hay con qué comparar.

Requiere, en la misma carpeta (o ajusta ASSETS_DIR abajo):
    - modelo_S4_lstm.keras
    - SCALER_S4_app.pkl
    - config.json
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="Pronóstico S4 Ionosférico", layout="wide")

ASSETS_DIR = Path(__file__).parent
MODEL_PATH = ASSETS_DIR / "modelo_S4_lstm.keras"
SCALER_PATH = ASSETS_DIR / "SCALER_S4_app.pkl"
CONFIG_PATH = ASSETS_DIR / "config.json"


# =============================================================================
# 2. CARGA EN CACHÉ DE RECURSOS
# =============================================================================
@st.cache_resource
def load_assets():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    # Solo inferencia: no hace falta el optimizador ni el loss personalizado.
    model = load_model(MODEL_PATH, compile=False)

    features_cols = config["FEATURES_COLS"]
    target_col = config["TARGET_COL"]
    idx_target = features_cols.index(target_col)
    min_target = scaler.data_min_[idx_target]
    max_target = scaler.data_max_[idx_target]

    return model, scaler, config, min_target, max_target


# =============================================================================
# 3. FEATURE ENGINEERING (idéntico al del entrenamiento -- verificado)
# =============================================================================
def add_temporal_features(df):
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    minute_of_day = idx.hour * 60 + idx.minute
    out["daily_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["daily_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    days_in_year = np.where(idx.is_leap_year, 366, 365)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (idx.dayofyear - 1) / days_in_year)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (idx.dayofyear - 1) / days_in_year)
    return out


def desnormalizar(values_scaled, min_v, max_v):
    return np.abs(values_scaled * (max_v - min_v) + min_v)


def verificar_continuidad(index, sampling_minutes):
    if len(index) < 2:
        return 0
    diffs = np.diff(index.values).astype("timedelta64[m]").astype(int)
    return int(np.sum(diffs != sampling_minutes))


# =============================================================================
# 4. LÓGICA DE EXPLORACIÓN (posiciones de corte, búsqueda de evento/calma)
# =============================================================================
def rango_posiciones_validas(n_filas, lookback, horizon):
    """pos_min: primera posición con LOOKBACK filas completas antes.
    pos_max_con_comparacion: última posición con HORIZON filas reales
    completas después (para poder comparar). Puede no haber ninguna."""
    pos_min = lookback - 1
    pos_max_sin_comparacion = n_filas - 1
    pos_max_con_comparacion = n_filas - horizon - 1
    return pos_min, pos_max_sin_comparacion, pos_max_con_comparacion


def future_max_por_posicion(target_array, horizon):
    """future_max[pos] = max(target[pos+1 : pos+1+horizon]). Vectorizado."""
    n = len(target_array)
    if n <= horizon:
        return np.array([])
    windows = np.lib.stride_tricks.sliding_window_view(target_array[1:], horizon)
    return windows.max(axis=1)


def buscar_evento_mas_fuerte(target_array, pos_min, pos_max_con_comparacion, horizon):
    future_max = future_max_por_posicion(target_array, horizon)
    if pos_max_con_comparacion < pos_min or len(future_max) <= pos_max_con_comparacion:
        return None
    tramo = future_max[pos_min : pos_max_con_comparacion + 1]
    if len(tramo) == 0:
        return None
    return pos_min + int(np.argmax(tramo))


def buscar_periodo_calmo(target_array, pos_min, pos_max_con_comparacion, horizon):
    future_max = future_max_por_posicion(target_array, horizon)
    if pos_max_con_comparacion < pos_min or len(future_max) <= pos_max_con_comparacion:
        return None
    tramo = future_max[pos_min : pos_max_con_comparacion + 1]
    if len(tramo) == 0:
        return None
    return pos_min + int(np.argmin(tramo))


def listar_eventos_detectados(target_array, index, threshold, horizon, pos_min,
                               pos_max_con_comparacion, max_eventos=150):
    """Detecta segmentos contiguos de evento (S4 >= threshold) y devuelve,
    por cada uno, la posición de corte justo ANTES de que empiece (para que
    el horizonte de predicción tenga que cubrirlo), su timestamp y su pico
    real. Si hay demasiados eventos para un desplegable usable, se queda
    con los `max_eventos` de mayor pico (y los reordena cronológicamente)."""
    es_evento = target_array >= threshold
    cambia = np.diff(es_evento.astype(int), prepend=0)
    inicios = np.where(cambia == 1)[0]
    fines = np.where(cambia == -1)[0] - 1
    if len(fines) < len(inicios):
        fines = np.append(fines, len(target_array) - 1)

    eventos = []
    for ini, fin in zip(inicios, fines):
        pos_corte_evento = ini - 1
        if pos_corte_evento < pos_min or pos_corte_evento > pos_max_con_comparacion:
            continue  # evento sin suficiente historia antes, o sin horizonte completo después
        pico = float(target_array[ini : fin + 1].max())
        eventos.append({"pos": int(pos_corte_evento), "inicio": index[ini], "pico": pico})

    eventos.sort(key=lambda e: e["inicio"])
    if len(eventos) > max_eventos:
        eventos = sorted(eventos, key=lambda e: -e["pico"])[:max_eventos]
        eventos.sort(key=lambda e: e["inicio"])
    return eventos


def graficar_overview(index, target_array, threshold, pos_corte, max_puntos=3000):
    """Mini-gráfica de TODA la serie (downsampled tomando el máximo por
    bucket, para no perder eventos angostos al comprimir), con los eventos
    marcados en rojo y una línea vertical en el punto de corte actual --
    da contexto visual antes de mover el slider a ciegas."""
    n = len(target_array)
    bucket = max(1, n // max_puntos)
    n_buckets = int(np.ceil(n / bucket))
    resto = n_buckets * bucket - n
    padded = np.concatenate([target_array, np.full(resto, np.nan)]) if resto > 0 else target_array
    matriz = padded.reshape(n_buckets, bucket)
    valores_bucket = np.nanmax(matriz, axis=1)
    pos_bucket = np.clip(np.arange(n_buckets) * bucket + bucket // 2, 0, n - 1)
    tiempos_bucket = index[pos_bucket]

    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.plot(tiempos_bucket, valores_bucket, color="#999999", linewidth=0.8)
    es_evt = valores_bucket >= threshold
    ax.scatter(np.array(tiempos_bucket)[es_evt], valores_bucket[es_evt], color="#FF4B4B", s=10, zorder=3)
    ax.axhline(threshold, color="#FFA500", linestyle=":", linewidth=1)
    ax.axvline(index[pos_corte], color="#3498DB", linewidth=1.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


# =============================================================================
# 5. PREDICCIÓN Y ARMADO DEL GRÁFICO COMPARATIVO
# =============================================================================
def predecir_en_posicion(df_fe, df_crudo, pos_corte, model, scaler, features_cols, target_col,
                          lookback, horizon, min_target, max_target, sampling_minutes=1):
    ventana = df_fe.iloc[pos_corte - lookback + 1 : pos_corte + 1][features_cols]
    ventana_scaled = scaler.transform(ventana)
    X_input = ventana_scaled.reshape(1, lookback, len(features_cols))
    y_pred_scaled = model.predict(X_input, verbose=0)[0]
    y_pred_real = desnormalizar(y_pred_scaled, min_target, max_target)

    cutoff_time = df_crudo.index[pos_corte]
    # Los instantes futuros se construyen por calendario: así el pronóstico
    # siempre tiene sus HORIZON marcas de tiempo, aunque el archivo termine
    # en el punto de corte (modo pronóstico puro) o tenga huecos después.
    idx_futuro = pd.date_range(cutoff_time + pd.Timedelta(minutes=sampling_minutes),
                               periods=horizon, freq=f"{sampling_minutes}min")
    return y_pred_real, cutoff_time, idx_futuro


def construir_grafico_comparativo(df_crudo, target_col, pos_corte, minutos_contexto,
                                   y_pred_real, cutoff_time, idx_futuro, horizon):
    idx_historia = df_crudo.index[max(0, pos_corte - minutos_contexto + 1) : pos_corte + 1]
    serie_obs = df_crudo[target_col].reindex(idx_historia)

    chart_df = pd.DataFrame({"S4 observado": serie_obs})

    # Solo se compara si TODOS los minutos pronosticados existen en el archivo.
    hay_comparacion = bool(pd.Index(idx_futuro).isin(df_crudo.index).all())
    serie_pred = pd.Series(y_pred_real, index=idx_futuro)
    conexion = pd.Series([serie_obs.iloc[-1]], index=[cutoff_time])
    serie_pred_c = pd.concat([conexion, serie_pred])
    chart_df = chart_df.reindex(chart_df.index.union(serie_pred_c.index))
    chart_df["S4 pronosticado"] = serie_pred_c.reindex(chart_df.index)

    rmse_ventana = None
    if hay_comparacion:
        serie_real = df_crudo[target_col].reindex(idx_futuro)
        serie_real_c = pd.concat([conexion, serie_real])
        chart_df = chart_df.reindex(chart_df.index.union(serie_real_c.index))
        chart_df["S4 real"] = serie_real_c.reindex(chart_df.index)
        rmse_ventana = float(np.sqrt(np.mean((serie_real.to_numpy() - serie_pred.to_numpy()) ** 2)))

    return chart_df.sort_index(), hay_comparacion, rmse_ventana


def graficar_pronostico(df_crudo, target_col, pos_corte, min_historia, min_posterior,
                        y_pred_real, idx_futuro, threshold, sampling_minutes=1):
    """Gráfico principal con escala FIJA en Y (0 a ≥1.0) para que la magnitud del
    pico se aprecie, banda del horizonte pronosticado, umbral y, opcionalmente,
    el S4 real posterior al horizonte como contexto (no es pronóstico)."""
    paso = pd.Timedelta(minutes=sampling_minutes)
    t0 = df_crudo.index[pos_corte]
    serie = df_crudo[target_col]
    historia = serie.loc[t0 - (min_historia - 1) * paso : t0]
    conexion = pd.Series([serie.iloc[pos_corte]], index=[t0])
    pronostico = pd.concat([conexion, pd.Series(y_pred_real, index=idx_futuro)])
    real_h = serie.reindex(idx_futuro).dropna()
    real_horizonte = pd.concat([conexion, real_h]) if len(real_h) else pd.Series(dtype=float)
    fin_h = idx_futuro[-1]
    posterior = serie.loc[fin_h : fin_h + min_posterior * paso] if min_posterior > 0 else pd.Series(dtype=float)

    nombres = ["S4 observado (historia)", "Pronóstico del modelo", "S4 real (horizonte)",
               "S4 real posterior (solo contexto)"]
    colores = ["#2ECC71", "#FF4B4B", "#3498DB", "#95A5A6"]
    partes = [historia, pronostico, real_horizonte, posterior]
    largo = pd.concat([pd.DataFrame({"Tiempo": s.index, "S4": s.to_numpy(), "Serie": n})
                       for n, s in zip(nombres, partes) if len(s)], ignore_index=True)
    y_max = max(1.0, float(np.nanmax(largo["S4"])) * 1.1)
    usados = [n for n, s in zip(nombres, partes) if len(s)]
    escala_color = alt.Scale(domain=usados, range=[c for n, c in zip(nombres, colores) if n in usados])

    banda = alt.Chart(pd.DataFrame({"ini": [t0], "fin": [fin_h]})).mark_rect(
        color="#3498DB", opacity=0.12).encode(x="ini:T", x2="fin:T")
    umbral = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="#F5A623", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    corte = alt.Chart(pd.DataFrame({"t": [t0]})).mark_rule(
        color="#7F8C8D", strokeWidth=1).encode(x="t:T")
    lineas = alt.Chart(largo).mark_line(strokeWidth=2).encode(
        x=alt.X("Tiempo:T", title="Hora (UT)", axis=alt.Axis(format="%d-%m %H:%M", labelAngle=0)),
        y=alt.Y("S4:Q", title="Índice S4", scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color("Serie:N", scale=escala_color, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("Tiempo:T", format="%Y-%m-%d %H:%M"), "Serie:N", alt.Tooltip("S4:Q", format=".3f")],
    )
    return (banda + umbral + corte + lineas).properties(height=420).interactive(bind_y=False)


def ventana_del_dia(t0, modo):
    """Inicio y fin del día que contiene t0.
    'ionosferico': de 12:00 UT a 11:59 UT del día siguiente (07:00 a 06:59 hora de Lima),
    así una misma noche de centelleo (19–02 LT = 00–07 UT) queda completa en un solo gráfico.
    'ut': día calendario UT (00:00 a 23:59 UT)."""
    if modo == "ionosferico":
        inicio = (t0 - pd.Timedelta(hours=12)).normalize() + pd.Timedelta(hours=12)
    else:
        inicio = t0.normalize()
    return inicio, inicio + pd.Timedelta(hours=24) - pd.Timedelta(minutes=1)


def graficar_dia(df_crudo, target_col, t0, idx_futuro, y_pred_real, threshold, modo_dia):
    """Segundo gráfico: el día completo que contiene el punto de corte, con el perfil
    medio del archivo (si cubre ≥ 3 días) para comparar con el comportamiento típico."""
    inicio, fin = ventana_del_dia(t0, modo_dia)
    serie = df_crudo[target_col]
    dia = serie.loc[inicio:fin]
    partes = [("S4 del día", dia)]
    n_dias = serie.index.normalize().nunique()
    if n_dias >= 3:
        minuto = serie.index.hour * 60 + serie.index.minute
        perfil = serie.groupby(minuto).mean()
        malla = pd.date_range(inicio, fin, freq="1min")
        tipico = pd.Series(perfil.reindex(malla.hour * 60 + malla.minute).to_numpy(), index=malla)
        partes.append((f"Perfil medio del archivo ({n_dias} días)", tipico.dropna()))
    partes.append(("Pronóstico del modelo", pd.Series(y_pred_real, index=idx_futuro)))
    largo = pd.concat([pd.DataFrame({"Tiempo": s.index, "S4": s.to_numpy(), "Serie": n})
                       for n, s in partes if len(s)], ignore_index=True)
    y_max = max(1.0, float(np.nanmax(largo["S4"])) * 1.1)
    dominio = [n for n, s in partes if len(s)]
    rango = {"S4 del día": "#2ECC71", "Pronóstico del modelo": "#FF4B4B"}
    escala = alt.Scale(domain=dominio, range=[rango.get(n, "#95A5A6") for n in dominio])

    banda = alt.Chart(pd.DataFrame({"ini": [t0], "fin": [idx_futuro[-1]]})).mark_rect(
        color="#3498DB", opacity=0.25).encode(x="ini:T", x2="fin:T")
    umbral = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="#F5A623", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    lineas = alt.Chart(largo).mark_line(strokeWidth=1.5).encode(
        x=alt.X("Tiempo:T", title="Hora (UT) · hora de Lima = UT − 5",
                scale=alt.Scale(domain=[inicio.isoformat(), fin.isoformat()]),
                axis=alt.Axis(format="%H:%M", labelAngle=0)),
        y=alt.Y("S4:Q", title="Índice S4", scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color("Serie:N", scale=escala, legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.Serie == dominio[1] if len(dominio) > 2 else alt.datum.Serie == "",
                                 alt.value([4, 3]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("Tiempo:T", format="%Y-%m-%d %H:%M"), "Serie:N", alt.Tooltip("S4:Q", format=".3f")],
    )
    resumen = {
        "inicio": inicio, "fin": fin, "max": float(dia.max()) if len(dia) else float("nan"),
        "t_max": dia.idxmax() if len(dia) else None,
        "min_evento": int((dia >= threshold).sum()),
        "primer_cruce": dia[dia >= threshold].index[0] if (dia >= threshold).any() else None,
        "cobertura": 100 * len(dia) / 1440,
    }
    return (banda + umbral + lineas).properties(height=320).interactive(bind_y=False), resumen


# =============================================================================
# 6. INTERFAZ
# =============================================================================
def main():
    st.title("🛰️ Pronóstico de Centelleo Ionosférico (S4)")

    try:
        model, scaler, config, min_target, max_target = load_assets()
    except Exception as e:
        st.error(
            f"❌ No se pudieron cargar los archivos del modelo ({e}). Verifica que "
            f"'modelo_S4_lstm.keras', 'SCALER_S4_app.pkl' y 'config.json' estén junto a esta app."
        )
        st.stop()

    LOOKBACK = config["LOOKBACK"]
    HORIZON = config["HORIZON"]
    THRESHOLD = config["THRESHOLD"]
    # Umbral de ALERTA sobre el pronóstico: el calibrado en validación si está
    # en config.json ("UMBRAL_ALERTA"); si no, el umbral físico de evento.
    UMBRAL_ALERTA = config.get("UMBRAL_ALERTA", THRESHOLD)
    FEATURES_COLS = config["FEATURES_COLS"]
    TARGET_COL = config["TARGET_COL"]
    SAMPLING_MINUTES = config["SAMPLING_MINUTES"]

    with st.sidebar:
        st.success("✅ Modelo y scaler cargados.")
        st.caption(
            f"Arquitectura: LSTM {config.get('ARQUITECTURA', '?')} "
            f"({config.get('UNITS1', '?')}/{config.get('UNITS2', '?')} unidades)"
        )
        st.caption(f"LOOKBACK={LOOKBACK} min · HORIZON={HORIZON} min")
        st.caption(f"Umbral de evento S4 ≥ {THRESHOLD} · umbral de alerta {UMBRAL_ALERTA}")
        st.caption(f"RMSE_Evento (test): {config.get('RMSE_EVENTO_TEST', float('nan')):.4f}")

    st.markdown(
        f"Sube un CSV con columnas **`Tiempo`** y **`{TARGET_COL}`**. Con al menos "
        f"**{LOOKBACK} filas** funciona en modo pronóstico puro. Si subes **más filas** "
        f"(por ejemplo, un tramo largo de tu conjunto de test), la app habilita el modo "
        f"explorador: podrás moverte por el tiempo y comparar el pronóstico contra el "
        f"valor real que efectivamente ocurrió."
    )

    uploaded_file = st.file_uploader("Archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    time_col = "Tiempo" if "Tiempo" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col is None or TARGET_COL not in df.columns:
        st.error(f"⚠️ El CSV debe tener una columna de fecha ('Tiempo') y la columna '{TARGET_COL}'.")
        st.stop()

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    if len(df) < LOOKBACK:
        st.error(f"⚠️ Se necesitan al menos {LOOKBACK} filas. El archivo tiene {len(df)}.")
        st.stop()

    df_fe = add_temporal_features(df)
    target_array = df[TARGET_COL].to_numpy()
    n_filas = len(df)

    pos_min, pos_max_sin_comp, pos_max_con_comp = rango_posiciones_validas(n_filas, LOOKBACK, HORIZON)
    modo_exploracion = pos_max_con_comp >= pos_min  # hay al menos un punto con futuro real conocido

    st.subheader(f"Archivo cargado: {n_filas:,} filas ({df.index.min()} → {df.index.max()})")

    # --- Selección del punto de corte ---
    key_slider = "pos_corte"
    # Al cambiar de archivo, el punto de corte guardado en la sesión pertenece al
    # archivo anterior y puede quedar fuera de rango (p. ej. 3109 en un archivo de
    # 400 filas). Se reinicia el estado al cargar un archivo distinto y, por
    # seguridad, se acota siempre al rango válido del archivo actual.
    id_archivo = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("archivo_actual") != id_archivo:
        st.session_state["archivo_actual"] = id_archivo
        st.session_state[key_slider] = pos_max_sin_comp  # por defecto: el final del archivo
        st.session_state.pop("selector_evento", None)
    if not (pos_min <= st.session_state.get(key_slider, -1) <= pos_max_sin_comp):
        st.session_state[key_slider] = pos_max_sin_comp

    if modo_exploracion:
        st.markdown("### 🔍 Modo explorador -- elige dónde \"cortar\" la historia")

        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("🔴 Ir al evento más fuerte disponible"):
            pos = buscar_evento_mas_fuerte(target_array, pos_min, pos_max_con_comp, HORIZON)
            if pos is not None:
                st.session_state[key_slider] = pos
        if col_b2.button("🟢 Ir a un período tranquilo"):
            pos = buscar_periodo_calmo(target_array, pos_min, pos_max_con_comp, HORIZON)
            if pos is not None:
                st.session_state[key_slider] = pos
        if col_b3.button("⏭️ Ir al final del archivo (sin comparación)"):
            st.session_state[key_slider] = pos_max_sin_comp

        # Desplegable con TODOS los eventos detectados (no solo el más
        # fuerte) -- cada uno salta al punto justo antes de que empiece.
        eventos = listar_eventos_detectados(target_array, df.index, THRESHOLD, HORIZON, pos_min, pos_max_con_comp)
        if eventos:
            opciones = ["-- elegir un evento específico --"] + [
                f"{e['inicio']:%Y-%m-%d %H:%M} · pico S4={e['pico']:.3f}" for e in eventos
            ]
            pos_por_opcion = {opciones[i + 1]: e["pos"] for i, e in enumerate(eventos)}

            def _saltar_a_evento_elegido():
                elegido = st.session_state["selector_evento"]
                if elegido in pos_por_opcion:
                    st.session_state[key_slider] = pos_por_opcion[elegido]

            st.selectbox(
                f"O elige entre los {len(eventos)} eventos detectados en el archivo:",
                options=opciones,
                key="selector_evento",
                on_change=_saltar_a_evento_elegido,
            )
        else:
            st.caption("No se detectaron eventos (S4 ≥ umbral) con suficiente contexto en este archivo.")

        # Mini-resumen visual: toda la serie, eventos en rojo, línea azul en
        # el punto de corte actual -- para no mover el slider a ciegas.
        st.pyplot(graficar_overview(df.index, target_array, THRESHOLD, st.session_state[key_slider]))

        pos_corte = st.slider(
            "Punto de corte (fin del historial usado por el modelo)",
            min_value=pos_min,
            max_value=pos_max_sin_comp,
            key=key_slider,
        )
        st.caption(f"Corte en: **{df.index[pos_corte]}**")
    else:
        pos_corte = pos_max_sin_comp
        st.info(
            "El archivo no tiene suficientes filas después del final para comparar contra "
            f"un valor real (se necesitarían {HORIZON} filas más) -- modo pronóstico puro."
        )

    col_s1, col_s2 = st.columns(2)
    minutos_contexto = col_s1.slider(
        "Minutos de historia a mostrar en el gráfico",
        min_value=30,
        max_value=LOOKBACK,
        value=min(180, LOOKBACK),
        step=10,
        help=(
            f"Solo afecta lo que se GRAFICA. El modelo siempre usa los {LOOKBACK} "
            "minutos completos para predecir."
        ),
    )
    minutos_posteriores = 0
    if modo_exploracion:
        minutos_posteriores = col_s2.slider(
            "Minutos de S4 real a mostrar después del horizonte",
            min_value=0,
            max_value=180,
            value=180,
            step=10,
            help=(
                f"Solo CONTEXTO visual (línea gris): el modelo pronostica únicamente "
                f"{HORIZON} minutos. Sirve para ver cómo evolucionó el evento completo."
            ),
        )

    ventana_cruda = df.iloc[pos_corte - LOOKBACK + 1 : pos_corte + 1]
    n_huecos = verificar_continuidad(ventana_cruda.index, SAMPLING_MINUTES)
    if n_huecos > 0:
        st.warning(
            f"⚠️ {n_huecos} discontinuidad(es) en los {LOOKBACK} min usados como historia -- "
            "la predicción puede no ser confiable."
        )

    if st.button(f"Generar pronóstico a {HORIZON} minutos", type="primary"):
        with st.spinner("Calculando..."):
            y_pred_real, cutoff_time, idx_futuro = predecir_en_posicion(
                df_fe, df, pos_corte, model, scaler, FEATURES_COLS, TARGET_COL,
                LOOKBACK, HORIZON, min_target, max_target, SAMPLING_MINUTES,
            )
            chart_df, hay_comparacion, rmse_ventana = construir_grafico_comparativo(
                df, TARGET_COL, pos_corte, minutos_contexto, y_pred_real, cutoff_time, idx_futuro, HORIZON
            )

        st.subheader(f"Observado + pronóstico desde {cutoff_time:%Y-%m-%d %H:%M}")
        st.altair_chart(
            graficar_pronostico(df, TARGET_COL, pos_corte, minutos_contexto, minutos_posteriores,
                                y_pred_real, idx_futuro, THRESHOLD, SAMPLING_MINUTES),
            use_container_width=True,
        )
        st.caption(
            f"Banda azul: los {HORIZON} minutos pronosticados. Línea naranja: umbral S4 = {THRESHOLD}. "
            "Escala vertical fija desde 0 para comparar la magnitud entre eventos."
        )

        # --- Segundo gráfico: el día completo, para verificar el comportamiento típico ---
        st.subheader("Contexto del día completo")
        modo_dia = st.radio(
            "Día a mostrar",
            options=["ionosferico", "ut"],
            format_func=lambda m: ("Noche completa: 07:00 a 07:00 hora de Lima (recomendado)"
                                   if m == "ionosferico" else "Día calendario UT (00:00 a 24:00 UT)"),
            horizontal=True,
        )
        grafico_dia, r = graficar_dia(df, TARGET_COL, cutoff_time, idx_futuro, y_pred_real, THRESHOLD, modo_dia)
        st.altair_chart(grafico_dia, use_container_width=True)
        lima = pd.Timedelta(hours=5)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S4 máximo del día", f"{r['max']:.3f}",
                  help=f"A las {r['t_max']:%H:%M} UT ({(r['t_max'] - lima):%H:%M} Lima)" if r["t_max"] is not None else None)
        c2.metric("Minutos con S4 ≥ umbral", f"{r['min_evento']}")
        c3.metric("Primer cruce del umbral",
                  f"{(r['primer_cruce'] - lima):%H:%M} Lima" if r["primer_cruce"] is not None else "sin evento")
        c4.metric("Cobertura de datos", f"{r['cobertura']:.0f} %")
        st.caption(
            f"Periodo: {r['inicio']:%Y-%m-%d %H:%M} a {r['fin']:%Y-%m-%d %H:%M} UT. Banda azul: el pronóstico actual. "
            "Línea gris punteada: S4 medio por hora del día en todo el archivo (aparece si cubre al menos 3 días). "
            "Lo típico en Jicamarca es calma de día y centelleo tras la puesta del Sol, entre 19:00 y 02:00 hora de Lima."
        )

        max_pred = float(np.max(y_pred_real))
        alerta_predicha = max_pred >= UMBRAL_ALERTA

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Máximo pronosticado", f"{max_pred:.3f}")
        if hay_comparacion:
            max_real = float(df[TARGET_COL].reindex(idx_futuro).max())
            alerta_real = max_real >= THRESHOLD
            col_m2.metric("Máximo real", f"{max_real:.3f}")
            col_m3.metric("RMSE de la ventana", f"{rmse_ventana:.4f}")

            if alerta_predicha == alerta_real:
                st.success(
                    f"✅ Coincidencia: el modelo {'sí' if alerta_predicha else 'no'} predijo evento, "
                    f"y {'sí' if alerta_real else 'no'} ocurrió realmente."
                )
            elif alerta_predicha and not alerta_real:
                st.warning("⚠️ Falso positivo: el modelo predijo evento, pero no ocurrió.")
            else:
                st.error("❌ Falso negativo: ocurrió un evento real que el modelo no predijo.")
        else:
            col_m2.metric("Máximo real", "no disponible")
            if alerta_predicha:
                st.warning(f"⚠️ **ALERTA DE EVENTO:** pronóstico máximo {max_pred:.4f} ≥ umbral de alerta {UMBRAL_ALERTA}.")
            else:
                st.info(f"✅ Condiciones normales pronosticadas (máximo {max_pred:.4f}, umbral de alerta {UMBRAL_ALERTA}).")

        tabla_pred = pd.DataFrame({"Timestamp": idx_futuro, "S4 pronosticado": y_pred_real})
        if hay_comparacion:
            tabla_pred["S4 real"] = df[TARGET_COL].reindex(idx_futuro).to_numpy()
            tabla_pred["Error absoluto"] = (tabla_pred["S4 pronosticado"] - tabla_pred["S4 real"]).abs()
        st.subheader("Detalle numérico")
        st.dataframe(tabla_pred.set_index("Timestamp"))


if __name__ == "__main__":
    main()
Aquí tienes una estructura profesional y lista para usar en tu archivo README.md:

Markdown
# 🛰️ Sistema de Predicción de Centelleo Ionosférico ($S_4$) con LSTM

Aplicación web desarrollada con **Streamlit** y **TensorFlow/Keras** para el pronóstico multipaso del índice de centelleo ionosférico ($S_4$) a 10 minutos en el futuro, utilizando secuencias de entrada de 70 minutos históricos.

---

## 📌 Características Principales

* **Focal MSE Loss:** Carga dinámica de una función de pérdida personalizada diseñada para priorizar eventos severos de centelleo ($S_4 \ge 0.6$) y penalizar subestimaciones.
* **Feature Engineering Cíclico:** Generación automática de variables trigonométricas temporales (seno/coseno diarios y anuales).
* **Escalamiento Aislado:** Carga de `SCALER_S4.pkl` entrenado exclusivamente sobre conjuntos históricos para evitar fugas de datos.
* **Visualización Continua:** Gráfico de doble serie temporal que conecta en tiempo real los 70 minutos de observación pasada (verde) con los 10 minutos pronosticados (rojo).
* **Garantía Física:** Conversión a escala real mediante valor absoluto $\mid S_4 \mid$ para evitar predicciones negativas inconsistentes.
* **Sistema de Alerta de Eventos:** Advertencia visual automática en caso de detectar umbrales de centelleo de riesgo.

---

## 📁 Estructura del Proyecto

```text
├── app.py                                # Código principal de la aplicación Streamlit
├── SCALER_S4.pkl                        # Scaler serializado (MinMaxScaler/StandardScaler)
├── modelo_lstm_experimento_S4_1808.keras # Modelo LSTM entrenado
├── requirements.txt                      # Dependencias del proyecto
└── README.md                             # Documentación del proyecto

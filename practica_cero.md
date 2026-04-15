# Práctica Cero: Anatomía y Análisis de un Dataset

**Autor:** Eduardo Alcaraz  
**Fecha:** 2026-04-15  
**Entorno:** Jupyter Notebook / Python 3

---

## Objetivo

Desarrollar criterio analítico para diferenciar entre la **obtención pasiva de datos** (búsqueda), la **instrumentación activa** (creación) y el **Análisis Exploratorio de Datos (EDA)**, comprendiendo por qué esta fase es irremplazable antes de entrenar cualquier modelo predictivo.

---

## Fase 1: Análisis de Casos — Búsqueda vs. Creación

### Escenario A — Búsqueda e Integración: Predicción de Demanda Eléctrica en Michoacán

**Contexto:** Los datos ya existen públicamente (reportes CENACE, bases meteorológicas del SMN, etc.). El problema no es generarlos, sino **integrarlos y limpiarlos**.

**Retos principales:**

- **Data wrangling:** Los reportes del CENACE pueden venir en formatos distintos (CSV, PDF, XLS) con esquemas inconsistentes entre períodos.
- **Alineación temporal:** Cruzar lecturas de temperatura (que pueden estar en resolución horaria) con consumo eléctrico en MW/h (resolución de 15 min o diaria) requiere remuestreo cuidadoso.
- **Valores nulos (NaN):** Estaciones meteorológicas tienen huecos por mantenimiento; sensores de red pueden fallar. Rellenar con interpolación lineal o forward-fill sin contexto puede introducir sesgo.
- **Resolución temporal inconsistente:** Datos históricos de antes de 2020 pueden tener granularidad diferente a los actuales.

**Conclusión del escenario:** La calidad del modelo final depende casi en su totalidad de la calidad del proceso de limpieza. Un dataset con NaN mal tratados puede hacer que un modelo excelente produzca predicciones pésimas.

---

### Escenario B — Creación e Instrumentación: Comportamiento Térmico bajo Soldadura

**Contexto:** No existe ningún dataset en internet para la pieza metálica específica bajo estudio. Los datos deben **crearse desde cero mediante instrumentación física**.

**Retos principales:**

- **Selección de hardware:** Elegir el tipo correcto de termopar (tipo K para rangos de soldadura, hasta ~1260 °C) y el sistema DAQ (Data Acquisition System) con la resolución adecuada.
- **Frecuencia de muestreo:** Debe ser suficientemente alta para capturar transitorios térmicos (Nyquist), pero no tan alta que genere volúmenes inmanejables de datos.
- **Ruido electromagnético (EMI):** Las máquinas de soldar generan interferencia electromagnética severa que puede saturar lecturas o introducir pulsos ficticios en la señal.
- **Fidelidad física del dato:** El montaje del termopar (contacto directo, pasta térmica, distancia al arco) determina si lo que se mide refleja realmente la temperatura de la pieza.

**Conclusión del escenario:** En instrumentación, el ruido no es un problema estadístico abstracto — tiene una causa física identificable. Entenderla es prerequisito para cualquier limpieza posterior.

---

## Fase 2: Análisis Exploratorio de Datos (EDA) — Sanity Check

### Script de inspección con anomalías intencionales

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. GENERACIÓN DE DATOS
#    Señal base: tendencia lineal + oscilación sinusoidal + ruido gaussiano
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
tiempo = pd.date_range(start='2026-04-15', periods=500, freq='T')

valores = (
    np.linspace(10, 50, 500)          # Tendencia ascendente
    + np.sin(np.linspace(0, 20, 500)) * 5   # Oscilación periódica
    + np.random.normal(0, 2, 500)            # Ruido gaussiano
)

df = pd.DataFrame({'Timestamp': tiempo, 'Lectura': valores})
df.set_index('Timestamp', inplace=True)

# ─────────────────────────────────────────────────────────────
# 2. INYECCIÓN DE ANOMALÍAS INTENCIONALES
#    (Modificación del alumno — simula fallos reales de sensor)
# ─────────────────────────────────────────────────────────────

# Pico positivo (spike de ruido EMI o saturación del ADC)
df.iloc[120, 0] = 95.0

# Caída negativa abrupta (pérdida de contacto del termopar)
df.iloc[250, 0] = -10.0

# Segmento plano (sensor congelado / fallo de comunicación serial)
df.iloc[380:395, 0] = df.iloc[379, 0]

# Valor NaN (paquete perdido en transmisión)
df.iloc[460, 0] = np.nan

# ─────────────────────────────────────────────────────────────
# 3. INSPECCIÓN TEMPORAL — ¿Hay valores atípicos o estacionalidad?
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Lectura'], label='Señal Cruda del Sensor',
         color='#2ca02c', linewidth=0.8)
plt.axhline(df['Lectura'].mean(), color='red', linestyle='--',
            linewidth=1, label=f'Media = {df["Lectura"].mean():.2f}')
plt.title("Inspección Visual de la Serie de Tiempo (con anomalías inyectadas)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("serie_tiempo.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────
# 4. ANÁLISIS DE DISTRIBUCIÓN — ¿El ruido es normal?
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
sns.histplot(df['Lectura'].dropna(), kde=True, bins=35,
             color='steelblue', edgecolor='white')
plt.title("Histograma y Densidad de las Lecturas\n(distribución esperada: Gaussiana + tendencia)")
plt.xlabel("Amplitud de Lectura")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("histograma.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────
# 5. ESTADÍSTICA DESCRIPTIVA — El perfil completo del dataset
# ─────────────────────────────────────────────────────────────
print("=" * 45)
print("   PERFIL ESTADÍSTICO DEL DATASET")
print("=" * 45)
print(df.describe())
print(f"\nValores nulos: {df['Lectura'].isna().sum()}")
print(f"Rango intercuartil (IQR): {df['Lectura'].quantile(0.75) - df['Lectura'].quantile(0.25):.3f}")
print(f"Asimetría (skewness):     {df['Lectura'].skew():.3f}")
print(f"Curtosis (kurtosis):      {df['Lectura'].kurt():.3f}")
```

---

## Fase 3: Entregable y Conclusiones Críticas

### Pregunta 1 — ¿La serie de tiempo es estacionaria?

**Observación de la gráfica temporal:**

La serie **no es estacionaria**. Se puede observar claramente una **tendencia ascendente** (de ~10 a ~50 unidades en los 500 puntos de tiempo), lo que significa que la **media cambia con el tiempo**. Una señal estacionaria debería oscilar alrededor de un valor constante con varianza constante.

**¿Por qué es un factor crítico antes de entrenar?**

La estacionariedad es un supuesto fundamental en muchos modelos de series de tiempo (ARIMA, modelos con ventanas deslizantes, incluso redes LSTM si no se normaliza correctamente). Si una serie tiene tendencia:

- El modelo intentará "memorizar" la pendiente en lugar de aprender patrones de interés real (oscilaciones, estacionalidad).
- Los datos de entrenamiento y prueba tendrán distribuciones distintas: la pérdida de validación será engañosa.
- Las predicciones se sesgarán sistemáticamente hacia los valores del período de entrenamiento.

**Solución práctica:** Aplicar diferenciación (`df.diff()`), transformación logarítmica o descomposición STL para aislar la tendencia antes de ingresar la señal a cualquier modelo.

---

### Pregunta 2 — Factores físicos que generan outliers (Escenario B)

Si yo hubiera construido el circuito de captura para este dataset, los siguientes fenómenos físicos podrían producir picos atípicos en las lecturas:

| Causa física | Tipo de anomalía | Firma en la señal |
|---|---|---|
| **Ruido EMI del arco de soldadura** | Spike instantáneo | Pico positivo o negativo de duración < 1 muestra |
| **Pérdida de contacto del termopar** | Caída abrupta o lectura al infinito | Salto a valor muy bajo o muy alto y recuperación brusca |
| **Saturación del ADC** | Clipping | Señal "plana" en el valor máximo del rango |
| **Microcorte de alimentación del DAQ** | Segmento plano o NaN | Bloque de muestras idénticas o ausentes |
| **Soldadura heterogénea (salpicadura)** | Transitorio real | Pico de temperatura físicamente válido pero inusual |
| **Diferencia de tierra (ground loop)** | Ruido oscilatorio de 60 Hz | Modulación periódica superpuesta a la señal |
| **Variación de conductividad por oxidación del termopar** | Drift lento | Desviación creciente de la línea base |

**Nota crítica:** No todos los outliers son errores de medición. Un pico de temperatura real durante una salpicadura de soldadura es un evento físico legítimo que no debe eliminarse sin comprensión del fenómeno. Esta distinción solo la puede hacer el ingeniero que conoce el proceso.

---

### Pregunta 3 — Riesgos de conectar datos crudos directamente a una Red Neuronal

Saltarse el EDA y alimentar datos crudos a una red neuronal implica los siguientes riesgos concretos:

**1. Contaminación de gradientes por outliers**  
Un único valor de 95.0 en una señal de rango 10–50 puede dominar la función de pérdida (MSE es sensible al cuadrado del error), distorsionando los pesos en toda la red durante backpropagation.

**2. Normalización inválida**  
Si se normaliza con `MinMaxScaler` sobre datos con outliers, toda la señal "normal" quedará comprimida en un rango muy estrecho (ej. 0.05 a 0.55), perdiendo resolución informativa.

**3. El modelo aprende el artefacto, no el fenómeno**  
Una red entrenada con segmentos planos (sensor congelado) aprenderá a predecir "no cambio" en ciertos contextos, comportamiento que no existe en el sistema real.

**4. Métricas de validación engañosas**  
Si los NaN son reemplazados automáticamente con ceros (comportamiento por defecto en algunos pipelines), el modelo aprenderá que ciertos instantes corresponden a lecturas de cero, corrompiendo patrones temporales.

**5. Imposibilidad de depuración post-hoc**  
Sin un EDA documentado, cuando el modelo en producción falla, es imposible saber si el error viene de la arquitectura, de los hiperparámetros o de los datos mismos. El EDA es la línea base que permite aislar cada variable.

**Conclusión general:**  
El EDA no es una formalidad académica. Es la única herramienta que garantiza que el ingeniero entiende lo que está entrenando antes de hacerlo. En contextos industriales (control térmico, predicción de demanda, mantenimiento predictivo), un modelo entrenado sobre datos sucios puede tomar decisiones con consecuencias físicas y económicas reales.

---

## Apéndice: Perfil estadístico esperado con anomalías inyectadas

```
=== PERFIL ESTADÍSTICO DEL DATASET ===

           Lectura
count   499.000000      ← 1 NaN excluido del conteo
mean     31.417...      ← Media desplazada por outliers
std      11.563...      ← Desviación estándar inflada por picos
min     -10.000000      ← Anomalía inyectada (caída)
25%      21.8...
50%      31.2...        ← Mediana más robusta que la media
75%      40.8...
max      95.000000      ← Anomalía inyectada (spike)

Valores nulos: 1
Rango intercuartil (IQR): ~19.0
Asimetría (skewness): ~0.15   ← Distribución aproximadamente simétrica
Curtosis (kurtosis):  ~2.8    ← Colas algo más pesadas de lo normal (leptocúrtica)
```

> **Interpretación:** Si el `max` o `min` se alejan significativamente del rango esperado físicamente (en este caso 10–50 unidades), es una señal inmediata de que existen datos que requieren investigación antes de continuar con cualquier modelado.

# Análisis Exploratorio de Datos (EDA) – Juego Bala + Salto + MLP

## 1. Introducción

Este análisis explora los datos generados en el juego **Bala + Salto + MLP**, cuyo objetivo es entrenar un modelo que prediga cuándo el jugador debe realizar una acción (saltar o agacharse) para esquivar una bala que se dirige hacia él.

Aspectos clave:

- Cada tiro tiene **velocidad y altura variables**, lo que genera una mayor diversidad en los datos.
- La bala nunca sobrepasa la altura máxima del jugador, por lo que las decisiones siempre son posibles (saltar o agacharse).
- Cada frame donde la bala se acerca al jugador se registra como un ejemplo.
- Se analizan las relaciones entre velocidad de la bala, distancia al jugador y la acción que realiza el jugador.
- Se consideran los efectos del **bias** en el modelo MLP.

---

## 2. Estructura del Dataset

El dataset se guarda en formato CSV con las siguientes columnas:

| Columna           | Tipo     | Descripción |
|------------------|---------|------------|
| velocidad_bala   | float   | Velocidad de la bala en píxeles por frame (negativa, ya que va de derecha a izquierda). |
| distancia        | float   | Distancia horizontal entre el jugador y la bala en el frame actual. |
| accion           | int     | Acción del jugador: 0 no hacer nada, 1 saltar, 2 agacharse. |

---

## 3. Tipos de Entradas y Salidas

**Entradas (Features):**

- velocidad_bala (float)
- distancia (float)

**Salida (Target):**

- accion (int: 0, 1 o 2)

Estas variables permiten entrenar un **clasificador supervisado**, como un MLP, que puede predecir la acción que el jugador debe realizar dependiendo de la situación.

---

## 4. Estadísticas Descriptivas

**Velocidad de la bala (velocidad_bala)**

- Rango: -12 a -6 píxeles/frame.
- Promedio: aproximadamente -9 píxeles/frame.
- Distribución: uniforme, ya que cada tiro se genera con valores aleatorios.

**Distancia al jugador (distancia)**

- Rango: desde el margen hasta la posición inicial de la bala (~100 a ~1000 píxeles).
- Promedio: alrededor de la mitad del recorrido de la bala.
- Distribución: continua, ya que la bala se va acercando progresivamente al jugador.

**Acción del jugador (accion)**

- 0: no realizar acción
- 1: saltar
- 2: agacharse

Esta variable representa un problema de **clasificación multiclase**, lo que hace el modelo más completo.

---

## 5. Distribución y Patrones

**Distribución de accion:**

- La cantidad de cada acción depende del estilo del jugador.
- Es importante tener datos balanceados entre las tres acciones (0, 1, 2).
- Si una acción predomina demasiado, el modelo puede sesgarse.

**Relación velocidad vs distancia:**

- La decisión depende de ambas variables:
  - Velocidad alta: menos tiempo de reacción → acciones rápidas.
  - Velocidad baja: más tiempo → decisiones más anticipadas.

**Altura del tiro:**

- La bala tiene altura variable dentro del rango del jugador.
- Si la bala va baja → el jugador salta.
- Si la bala va más alta → el jugador se agacha.
- La bala nunca sobrepasa la altura del jugador.

---

## 6. Visualizaciones

### 6.1 Gráfica 2D: Distancia vs Velocidad

![Gráfica 2D](imagenes/grafica_2d.png)

**Descripción:**

- Eje X: distancia jugador-bala
- Eje Y: velocidad de la bala
- Color: representa la acción (saltar, no hacer nada o agacharse)
- Permite identificar zonas donde ocurren ciertas decisiones.

---

### 6.2 Gráfica 3D: Distancia vs Velocidad vs Frame

![Gráfica 3D](imagenes/grafica_3d.png)

**Descripción:**

- Eje X: distancia
- Eje Y: velocidad
- Eje Z: frame (tiempo)
- Color: acción del jugador
- Permite analizar cómo evolucionan las decisiones en el tiempo.

---

## 7. Función del Bias en el MLP

- El bias permite ajustar la activación de las neuronas sin depender completamente de las entradas.
- Ayuda a tomar decisiones aunque los datos no sean exactos.
- En modelos simples:
  - Puede provocar que siempre se prediga la misma acción.
- En modelos bien entrenados:
  - Mejora la separación de clases y la precisión.

---

## 8. Recomendaciones para el Entrenamiento

1. Registrar las tres acciones (0, 1, 2).
2. Generar tiros con diferentes velocidades y alturas.
3. Registrar múltiples frames por tiro.
4. Evitar datos repetitivos.
5. Normalizar con StandardScaler antes de entrenar.

---

## 9. Conclusiones

- La variación en velocidad y altura mejora la calidad del dataset.
- La acción de agacharse hace el modelo más completo (multiclase).
- Las variables velocidad_bala y distancia son suficientes para modelar decisiones.
- El bias es importante para ajustar el comportamiento del modelo.
- Las visualizaciones ayudan a entender patrones.
- Con datos balanceados, el modelo puede aprender comportamientos similares a un jugador real.
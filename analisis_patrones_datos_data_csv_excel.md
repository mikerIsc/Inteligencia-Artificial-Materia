# Reporte de Análisis de Datos — `data.csv`

---

## Datos generales

- **Registros:** 150 filas, 4 columnas numéricas (`var_1`, `var_2`, `var_3`, `var_4`)
- **Valores nulos:** ninguno
- **Fecha:** 16/02/2026
- **Tipo de datos:** todos `float64`

---

## Estadística descriptiva

| | var_1 | var_2 | var_3 | var_4 |
|---|---|---|---|---|
| Media | 5.84 | 3.06 | 3.76 | 1.20 |
| Desv. Est. | 0.83 | 0.44 | **1.77** | 0.76 |
| Mínimo | 4.30 | 2.00 | 1.00 | 0.10 |
| Máximo | 7.90 | 4.40 | 6.90 | 2.50 |

`var_3` tiene la desviación estándar más alta, lo que indica que sus valores están muy dispersos — primera pista de que hay subgrupos en los datos.

---

## Patrón encontrado: tres grupos naturales

Al analizar la distribución de `var_3` se detecta una separación clara en **tres grupos**:

| Grupo | Rango de var_3 | Registros |
|---|---|---|
| Grupo A | 1.0 – 2.5 | 50 |
| Grupo B | 2.5 – 4.5 | 37 |
| Grupo C | 4.5 – 6.9 | 63 |

Cada grupo tiene un perfil distinto y consistente en todas las variables:

| Grupo | var_1 | var_2 | var_3 | var_4 |
|---|---|---|---|---|
| A | 5.01 | 3.43 | 1.46 | 0.25 |
| B | 5.72 | 2.70 | 4.07 | 1.28 |
| C | 6.58 | 2.98 | 5.40 | 1.91 |

De A → B → C todas las variables crecen progresivamente (excepto `var_2` que baja ligeramente). Los grupos son reales, no aleatorios.

---

## Correlaciones clave

| Par | Correlación | Interpretación |
|---|---|---|
| var_3 ↔ var_4 | **+0.963** | Crecen casi en proporción directa |
| var_1 ↔ var_3 | **+0.872** | Relación fuerte |
| var_1 ↔ var_4 | **+0.818** | Relación fuerte |
| var_2 ↔ var_3 | −0.428 | Relación débil inversa |
| var_2 ↔ var_1 | −0.118 | Sin relación relevante |

`var_2` es la variable más independiente del patrón. `var_3` y `var_4` son casi intercambiables entre sí.

---

## Outliers

Solo `var_2` presenta 4 valores fuera del rango esperado (IQR). No son críticos y no afectan el análisis general.

---

## Conclusiones

1. El dataset contiene **tres grupos diferenciados**, identificables principalmente por `var_3`.
2. Las variables más informativas son `var_3` y `var_4` por su alta correlación y capacidad de separar grupos.
3. `var_2` aporta poca información al patrón principal.
4. El dataset está limpio y listo para modelado sin preprocesamiento mayor.
5. Se recomienda confirmar con la fuente original qué representa cada variable antes de tomar decisiones sobre el dominio.

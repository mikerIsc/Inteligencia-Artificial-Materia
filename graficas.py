import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar dataset
df = pd.read_csv("datos_mlp.csv")

# ----------- GRÁFICA 2D -----------
plt.figure()

scatter = plt.scatter(
    df["distancia"],
    df["velocidad_bala"],
    c=df["salto"]
)

plt.xlabel("Distancia")
plt.ylabel("Velocidad de la bala")
plt.title("Distancia vs Velocidad")

plt.savefig("imagenes/grafica_2d.png")
plt.close()

# ----------- GRÁFICA 3D -----------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frames = range(len(df))

ax.scatter(
    df["distancia"],
    df["velocidad_bala"],
    frames,
    c=df["salto"]
)

ax.set_xlabel("Distancia")
ax.set_ylabel("Velocidad")
ax.set_zlabel("Frame")

plt.title("Distancia vs Velocidad vs Frame")

plt.savefig("imagenes/grafica_3d.png")
plt.close()

print("Gráficas generadas correctamente")
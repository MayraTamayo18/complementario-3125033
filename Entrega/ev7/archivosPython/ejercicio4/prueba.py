import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Cargar los datos desde un archivo CSV
df = pd.read_csv('archivo.csv')

# Asegúrate de que no hay valores NaN en las columnas de interés
df_clean = df.dropna(subset=['java'])

# Variables
X = np.arange(len(df_clean)).reshape(-1, 1)
y = df_clean['java'].values

# Modelo de regresión lineal
modelo = LinearRegression().fit(X, y)
y_pred = modelo.predict(X)

# Calcular coeficiente de correlación
correlacion, p_value = stats.pearsonr(X.flatten(), y)

# Hipótesis: si el valor p es menor que 0.05, rechazamos la hipótesis nula
# y concluimos que hay una relación significativa
if p_value < 0.05:
    print(f"Existe una relación significativa entre Semana y Java (p-value = {p_value:.4f}).")
    
    # Graficar los puntos y la línea de regresión
    plt.figure(figsize=(14, 8))  # Aumentar el tamaño de la figura
    plt.scatter(df_clean['Semana'], y, color='blue', s=100, label='Datos')
    plt.plot(df_clean['Semana'], y_pred, color='red', label='Línea de regresión')
    plt.title('Popularidad de Java a lo Largo del Tiempo', fontsize=16)
    plt.xlabel('Semana', fontsize=14)
    plt.ylabel('Java', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Ajustar la frecuencia de las etiquetas del eje X
    plt.xticks(df_clean['Semana'][::10], rotation=45)  # Mostrar una etiqueta cada 10 semanas
    
    plt.tight_layout()
    plt.show()
else:
    print(f"No existe una relación significativa entre Semana y Java (p-value = {p_value:.4f}).")
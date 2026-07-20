# ---------------- Importaciones

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from IPython.display import display

# ---------------- Fin de las Importaciones

# 1. Carga de datos

df = pd.read_csv(r'C:\Users\sasor\Desktop\Tripleten\Sprint 14\Proyecto\insurance_us.csv')

# 2. Verifica que los datos no tengan problemas: no faltan datos, no hay valores extremos, etc. 

"""primero para revisar"""
print(df.sample(10))
print(df.info())
print(df.dtypes)
print(df.isnull().sum())

df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})

"""se pasa age de float a int, ya que su naturaleza es: entero de anios, no hay decimales ni fracciones."""
df["age"] = df["age"].astype(int)
print(df["age"].dtype)

print(df.describe())

"""para comprobar"""
print(df.dtypes)
print(df.sample(10))

"""no hay nulos en ninguna columna"""

# 3. Analisis exploratorio de datos

g = sns.pairplot(df, kind="hist")
g.fig.set_size_inches(12, 12)
plt.show()

"""es dificil detectar grupos obvios viendo cada tabla con distintas combinaciones de variables por separado, porque la similitud entre clientes depende de todas. por eso conviene usar (kNN) que combine todas las caracteristicas en una sola metrica de similitud."""

# ------------------------ Tarea 1. Clientes similares

feature_names = ["gender", "age", "income", "family_members"]

def get_knn(df, n, k, metric):
    """
    Devuelve los k vecinos mas cercanos
    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar
    :param n: numero de objetos para los que se buscan los vecinos mas cercanos
    :param k: numero de vecinos mas cercanos a devolver
    :param metric: nombre de la metrica de distancia
    """
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)

    df_res = pd.concat([
        df.iloc[nbrs_indices[0]],
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=["distance"])
        ], axis=1)

    return df_res
 
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())
df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())

print(df_scaled.sample(5))

print("--- sin escalar, distancia euclidiana ---")
print(get_knn(df, 0, 5, "euclidean"))

print("--- escalado (MaxAbsScaler), distancia euclidiana ---")
print(get_knn(df_scaled, 0, 5, "euclidean"))

print("--- sin escalar, distancia manhattan ---")
print(get_knn(df, 0, 5, "manhattan"))

print("--- escalado (MaxAbsScaler), distancia manhattan ---")
print(get_knn(df_scaled, 0, 5, "manhattan"))

"""¿el hecho de que los datos no esten escalados afecta al algoritmo kNN? si. income tiene una escala mucho mayor que age, family_members o gender (0 ow 1). como kNN calcula distancias delos valores numericos, income termina: sin escalar, los vecinos mas cercanos casi siempre tienen el mismo salario exacto, sin importar edad, sexo o familiares. al escalar con MaxAbsScaler las 4 caracteristicas quedan en un rango comparable [0,1] y todas contribuyen de forma proporcional a la distancia."""

"""¿que tan similares son los resultados con manhattan (independiente del escalado)?
son parecidos a la distancia euclidiana. con pocas caracteristicas y datos continuos, ambas metricas terminan ordenando a los clientes de forma muy parecida pero la diferencia se nota mas en el valor exacto de la distancia que en el conjunto de vecinos elegido lo cual da por conclusion que el escalado es lo que mas cambia el resultado de knn, la metrica de distancia importa menos."""
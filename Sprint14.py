# ---------------- Importaciones

import math
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

# Tarea 2. ¿Es probable que el cliente reciba una prestacion del seguro?

df["insurance_benefits_received"] = (df["insurance_benefits"] > 0).astype(int)

print(df["insurance_benefits_received"].value_counts(normalize=True))

"""se confirma el desbalance: ~88.7% de los clientes no recibio ninguna prestacion (clase 0) y solo ~11.3% si (clase 1)."""

def eval_classifier(y_true, y_pred):
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f"F1: {f1_score:.2f}")
 
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all")
    print("Matriz de confusion")
    print(cm)

def rnd_model_predict(P, size, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)

for P in [0, df["insurance_benefits_received"].sum() / len(df), 0.5, 1]:
    print(f"la probabilidad: {P:.2f}")
    y_pred_rnd = rnd_model_predict(P, len(df))
 
    eval_classifier(df["insurance_benefits_received"], y_pred_rnd)
 
    print()

"""el modelo dummy con P=0 nunca predice prestacion, F1=0 con P=1 siempre predice prestacion, F1 bajo con 0.20 por muchos falsos positivos. con P=0.5 el F1 es similar (0.20). con P = 0.11 el F1 es apenas 0.12 esta es la base de comparacion para la tarea 2."""

X = df[feature_names]
y = df["insurance_benefits_received"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

print("--- knn clasificador, sin escalar ---")
for k in range(1, 11):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    print(f"k={k}: F1={f1:.3f}")

X_scaled = df_scaled[feature_names]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.3, random_state=12345)

print("--- knn clasificador, escalado ---")
for k in range(1, 11):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train_s)
    y_pred = knn.predict(X_test_s)
    f1 = sklearn.metrics.f1_score(y_test_s, y_pred)
    print(f"k={k}: F1={f1:.3f}")

"""¿puede un modelo entrenado funcionar mejor que un dummy? ¿puede funcionar peor?
si a ambas. sin escalar el knn entrenado va de F1=0.62  cuando k=1 , a casi 0 cuandoi k=10, degradandose conforme sube k, porque income domina la distancia igual que en la tarea 1... escalado, el knn entrenado llega a F1 entre 0.88 y 0.97, muy por encima de cualquier dummy"""

# Tarea 3. Regresion (con regresion lineal)

class MyLinearRegression:

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # anadir las unidades (columna de unos para el sesgo w0)
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)

    def predict(self, X):
        # anadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2.dot(self.weights)
        return y_pred

def eval_regressor(y_true, y_pred):
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")

    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print(f"R2: {r2_score:.2f}")
    
X = df[["age", "gender", "income", "family_members"]].to_numpy()
y = df["insurance_benefits"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()
lr.fit(X_train, y_train)
print("pesos (w0, w1, w2, w3, w4):", lr.weights)
y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)

X_scaled = df_scaled[["age", "gender", "income", "family_members"]].to_numpy()
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_scaled, y, test_size=0.3, random_state=12345)

lr_scaled = MyLinearRegression()
lr_scaled.fit(X_train_sc, y_train_sc)
print("pesos escalados (w0, w1, w2, w3, w4):", lr_scaled.weights)
y_test_pred_sc = lr_scaled.predict(X_test_sc)
eval_regressor(y_test_sc, y_test_pred_sc)

"""en ambos casos RMSE y R2 son identicos. los pesos si cambian pero eso se compensa con los pesos, asi que la prediccion final y la metrica de calidad quedan iguales"""


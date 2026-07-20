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

# Tarea 4. Ofuscar datos

personal_info_column_list = ["gender", "age", "income", "family_members"]
df_pn = df[personal_info_column_list]

X = df_pn.to_numpy()
print(X[:5])

rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))
print(P)

# comprobar que P sea invertible: si no lo fuera, np.linalg.inv lanzaria un error
P_inv = np.linalg.inv(P)
print(np.round(P @ P_inv, 6))

X_transformed = X @ P

print("datos originales (primeras 3 filas):")
print(X[:3])
print("datos transformados / ofuscados (primeras 3 filas):")
print(X_transformed[:3])

"""¿se puede adivinar la edad o los ingresos despues de la transformacion? no. cada columna de X @ P es una combinacion lineal de las 4 caracteristicas originales ponderada por numeros aleatorios, el resultado ya no se parece en escala ni magnitud a una edad o un salario real."""

X_recovered = X_transformed @ P_inv

print("datos recuperados / invertidos (primeras 3 filas):")
print(X_recovered[:3])
print("diferencia maxima absoluta entre original y recuperado:", np.max(np.abs(X - X_recovered)))

"""¿se pueden recuperar los datos originales conociendo P? si con X_transformado @ P_inv = X @ P @ P_inv = X @ I = X"""

"""--- 1. prueba de que la ofuscacion funciona con regresion lineal (demostracion analitica) ---
con datos originales: w = (X.T X)^-1 X.T y ; a = X w
ofuscando con X' = X P, el nuevo modelo da: w' = ((XP).T (XP))^-1 (XP).T y = (P.T X.T X P)^-1 P.T X.T y
usando (AB)^-1 = B^-1 A^-1 y (AB).T = B.T A.T:
w' = P^-1 (X.T X)^-1 (P.T)^-1 P.T X.T y = P^-1 (X.T X)^-1 X.T y = P^-1 w
entonces w' = P^-1 w.
prediccion con datos ofuscados: a' = X' w' = (X P)(P^-1 w) = X (P P^-1) w = X w = a
las predicciones a' son identicas a las predicciones a con datos originales. como y no cambia (solo se ofuscan las caracteristicas,
no el objetivo), el RMSE y R2 calculados sobre datos ofuscados deben ser exactamente iguales a los calculados sobre datos originales."""

def run_linear_regression(df, obfuscate=False, P=None, seed=12345):
    """
    entrena y evalua MyLinearRegression sobre insurance_benefits.
    si obfuscate=True, multiplica las caracteristicas por la matriz invertible P antes de entrenar.
    """
    X = df[personal_info_column_list].to_numpy().astype(float)
    y = df["insurance_benefits"].to_numpy()

    if obfuscate:
        X = X @ P
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    model = MyLinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
    r2 = sklearn.metrics.r2_score(y_test, y_pred)

    return y_pred, rmse, r2

y_pred_original, rmse_original, r2_original = run_linear_regression(df, obfuscate=False)
y_pred_ofuscado, rmse_ofuscado, r2_ofuscado = run_linear_regression(df, obfuscate=True, P=P)

print("--- datos originales ---")
print(f"RMSE: {rmse_original:.4f}")
print(f"R2: {r2_original:.4f}")

print("--- datos ofuscados (X @ P) ---")
print(f"RMSE: {rmse_ofuscado:.4f}")
print(f"R2: {r2_ofuscado:.4f}")

print("diferencia maxima absoluta entre predicciones:", np.max(np.abs(y_pred_original - y_pred_ofuscado)))


# ---- parte final

# # no, la ofuscacion no afecta en nada la calidad de la regresion lineal. si multiplico las caracteristicas por una matriz P invertible, lo unico que pasa es que el modelo ajusta sus pesos para compensar esa transformacion, los pesos nuevos terminan siendo P^-1 multiplicado por los pesos originales. como se compensa exactamente, las predicciones que da el modelo con los datos ofuscados son identicas a las que da con los datos originales, entonces el RECM tampoco cambia. por eso se puede ofuscar sin miedo a que el modelo empeore.

# # con los datos originales
# w = (X.T X)^-1 X.T y

# si ofusco las caracteristicas con X' = X @ P, el modelo entrenado con esos datos calcula otros pesos
# w_P = ((XP).T XP)^-1 (XP).T y

# aplico (AB).T = B.T A.T al termino (XP).T:
# w_P = (P.T X.T X P)^-1 P.T X.T y

# aplico (AB)^-1 = B^-1 A^-1 dos veces seguidas para separar la inversa del producto de matrices:
# (P.T X.T X P)^-1 = P^-1 (X.T X)^-1 (P.T)^-1

# sustituyendo:
# w_P = P^-1 (X.T X)^-1 (P.T)^-1 P.T X.T y

# como (P.T)^-1 P.T = I (matriz identidad), esos dos terminos se cancelan:
# w_P = P^-1 (X.T X)^-1 X.T y = P^-1 w

# entonces w_P = P^-1 * w, o sea que los pesos con datos ofuscados son los pesos originales multiplicados por P^-1.

# ahora reviso las predicciones con los datos ofuscados:
# a_P = (XP) w_P = (XP)(P^-1 w) = X (P P^-1) w = X (I) w = X w = a

# o sea que a_P es exactamente igual a (a)

# como (y) nunca se toca, solo se ofuscan las caracteristicas, y las predicciones son iguales, entonces el RECM calculado con los datos ofuscados tiene que dar el mismo valor que con los datos originales. la ofuscacion con una matriz P invertible no afecta la calidad del modelo de regresion lineal.

"""2. Prueba de regresión lineal con ofuscación de datos"""
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

# comprobar que P sea invertible: si no lo fuera, np.linalg.inv lanzaria un error
P_inv = np.linalg.inv(P)
print(np.round(P @ P_inv, 6))

def run_linear_regression(df, obfuscate=False, P=None, seed=12345):
    X = df[personal_info_column_list].to_numpy().astype(float)
    y = df["insurance_benefits"].to_numpy()

    if obfuscate:
        X = X @ P

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    model = MyLinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
    r2 = sklearn.metrics.r2_score(y_test, y_pred)

    return y_pred, rmse, r2


y_pred_original, rmse_original, r2_original = run_linear_regression(df, obfuscate=False)
"""datos originales"""
print(f"RMSE: {rmse_original:.4f}")
print(f"R2: {r2_original:.4f}")

y_pred_ofuscado, rmse_ofuscado, r2_ofuscado = run_linear_regression(df, obfuscate=True, P=P)
"""datos ofuscados (X @ P)"""
print(f"RMSE: {rmse_ofuscado:.4f}")
print(f"R2: {r2_ofuscado:.4f}")

print("diferencia entre predicciones:", np.max(np.abs(y_pred_original - y_pred_ofuscado)))
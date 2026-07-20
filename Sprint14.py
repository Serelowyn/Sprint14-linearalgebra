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

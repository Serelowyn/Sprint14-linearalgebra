# ---------------- Importaciones

import numpy as np
import pandas as pd

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

#2. Verifica que los datos no tengan problemas: no faltan datos, no hay valores extremos, etc.

# Renombramos las columnas para que el código se vea más coherente con su estilo.
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})

df.sample(10)

df.info()

# puede que queramos cambiar el tipo de edad (de float a int) aunque esto no es crucial

# escribe tu conversión aquí si lo deseas:

# comprueba que la conversión se haya realizado con éxito

# ahora echa un vistazo a las estadísticas descriptivas de los datos.
# ¿Se ve todo bien?


# 3. Análisis exploratorio de datos


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


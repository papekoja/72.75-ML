import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Cargar el conjunto de datos
data = pd.read_csv('german_credit.csv', sep=',', decimal='.')

#Dividir el conjunto de datos aleatoriamente en dos partes, el conjunto de entrenamiento y elconjunto de prueba
train, test = train_test_split(data, test_size=0.1, random_state=42)

#change all categorical variables to integer
train = train.apply(lambda x: pd.factorize(x)[0])
test = test.apply(lambda x: pd.factorize(x)[0])


#1. crear el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=40)



#2. entrenar el modelo
modelo.fit(train.drop('Creditability', axis=1), train['Creditability'])

#3. predecir
prediccion_train = modelo.predict(train.drop('Creditability', axis=1))
prediccion_test = modelo.predict(test.drop('Creditability', axis=1))

#4. calcular la precision
precision_train = sum(prediccion_train == train['Creditability']) / len(train) * 100
precision_test = sum(prediccion_test == test['Creditability']) / len(test) * 100

print('Precision del conjunto de entrenamiento: ', precision_train)
print('Precision del conjunto de prueba: ', precision_test)




#2. matriz de confusion para Random Forest
matriz_confusion = confusion_matrix(test['Creditability'], prediccion_test)
#en porcentaje
matriz_confusion = matriz_confusion / len(test) * 100

print('Matriz de confusion para Random Forest: \n', matriz_confusion)




#2. grafico de curvas de precision para Random Forest
"""
precision_train = []
precision_test = []
nodos = []

for i in range(1, 10):
    modelo = RandomForestClassifier(n_estimators=i, random_state=0)
    modelo.fit(train.drop('Creditability', axis=1), train['Creditability'])
    prediccion_train = modelo.predict(train.drop('Creditability', axis=1))
    prediccion_test = modelo.predict(test.drop('Creditability', axis=1))
    precision_train.append(sum(prediccion_train == train['Creditability']) / len(train) * 100)
    precision_test.append(sum(prediccion_test == test['Creditability']) / len(test) * 100)
    nodos.append(i)

plt.plot(nodos, precision_train, label='Conjunto de entrenamiento')

plt.plot(nodos, precision_test, label='Conjunto de prueba')

plt.xlabel('Cantidad de nodos')
plt.ylabel('Precision')
plt.title('Curva de precision para Random Forest')
plt.legend()
plt.show()
"""

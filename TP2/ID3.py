import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#Cargar el conjunto de datos
data = pd.read_csv('german_credit.csv', sep=',', decimal='.')

#Dividir el conjunto de datos aleatoriamente en dos partes, el conjunto de entrenamiento y elconjunto de prueba
train, test = train_test_split(data, test_size=0.1, random_state=40)

#change all categorical variables to integer
train = train.apply(lambda x: pd.factorize(x)[0])
test = test.apply(lambda x: pd.factorize(x)[0])

# function that divides the data into two parts, the training set and the test set, with a given percentage
def split_data(num_samples):
    # Use the 'sample' method to randomly select rows
    random_samples = data.sample(n=num_samples, random_state=42)
    train, test = train_test_split(random_samples, test_size=0.5, random_state=42)
    return train, test

#Implementar el algoritmo ID3 para clasificar los datos y poder determinar si una persona devolverá el crédito o no, utilizando todas las variables y la entropía de Shannon para la función Ganancia de Información.


#1. calcular la entropia de la variable objetivo
def entropia(data, target):
    #data: conjunto de datos
    #target: variable objetivo

    probabilidad = data[target].value_counts(normalize=True)
    entropia = sum(probabilidad * np.log2(probabilidad))
    return -entropia

#2. calcular la ganancia de informacion

def ganancia_informacion(data, atributo, target):

    #data: conjunto de datos
    #atributo: variable a analizar
    #target: variable objetivo

    entropia_total = entropia(data, target)
    valores_atributo = data[atributo].unique()
    entropia_atributo = 0

    for valor in valores_atributo:
        entropia_atributo += entropia(data[data[atributo] == valor], target) * len(data[data[atributo] == valor]) / len(data)

    ganancia = entropia_total - entropia_atributo
    return ganancia

#3. calcular la ganancia de informacion de todas las variables
def ganancia_informacion_total(data, target):
    #data: conjunto de datos
    #target: variable objetivo

    ganancia = {}

    for columna in data.drop(target, axis=1).columns:
        ganancia[columna] = ganancia_informacion(data, columna, target)

    return ganancia

#4. implementar el algoritmo ID3
def ID3(data, target, atributo_padre=None, valor_padre=None):

    #data: conjunto de datos
    #target: variable objetivo
    #atributo_padre: atributo padre
    #valor_padre: valor del atributo padre

    ganancia = ganancia_informacion_total(data, target)
    maximo = max(ganancia, key=ganancia.get)
    arbol = {maximo: {}}

    valores = np.unique(data[maximo])

    for valor in valores:
        subconjunto = data[data[maximo] == valor].reset_index(drop=True)
        valor_frecuente = subconjunto[target].value_counts().idxmax()

        if len(subconjunto[target].unique()) == 1:
            arbol[maximo][valor] = valor_frecuente
        else:
            arbol[maximo][valor] = ID3(subconjunto, target, maximo, valor_frecuente)
    return arbol

#5. predecir

def predecir(instancia, arbol, default=None):
        #instancia: instancia a predecir
        #arbol: arbol de decision
        #default: valor por defecto
    
        atributo = list(arbol.keys())[0]
        if instancia[atributo] in arbol[atributo].keys():
            resultado = arbol[atributo][instancia[atributo]]
            if isinstance(resultado, dict):
                return predecir(instancia, resultado)
            else:
                return resultado
        else:
            return default
        
#6. calcular la precision

def precision(data, arbol, target):
        
            #data: conjunto de datos
            #arbol: arbol de decision
            #target: variable objetivo
        
            data['prediccion'] = data.apply(predecir, axis=1, args=(arbol, 'No'))
            data['correcto'] = data.apply(lambda x: 1 if x[target] == x['prediccion'] else 0, axis=1)
            precision = data['correcto'].sum() / len(data) * 100
            return precision

#7. calcular la precision del conjunto de entrenamiento y del conjunto de prueba

arbol = ID3(train, 'Creditability')
precision_train = precision(train, arbol, 'Creditability')
precision_test = precision(test, arbol, 'Creditability')

print('Precision del conjunto de entrenamiento: ', precision_train)
print('Precision del conjunto de prueba: ', precision_test)

#1. matriz de confusion para ID3
test['prediccion'].fillna(0, inplace=True)
matriz_confusion = confusion_matrix(test['Creditability'], test['prediccion'])
matriz_confusion = matriz_confusion / len(test) * 100
print('Matriz de confusion para ID3: \n', matriz_confusion)

#1. grafico de curvas de precision para ID3

precision_train = []
precision_test = []
nodos = []

for i in list(range(100, 1001, 100)):
    train_local, test_local = split_data(i)
    arbol = ID3(train_local, 'Creditability')
    precision_train.append(precision(train_local, arbol, 'Creditability'))
    precision_test.append(precision(test_local, arbol, 'Creditability'))
    nodos.append(i)

plt.plot(nodos, precision_train, label='Conjunto de entrenamiento')

plt.plot(nodos, precision_test, label='Conjunto de prueba')

plt.xlabel('Cantidad de nodos')
plt.ylabel('Precision')
plt.title('Curva de precision para ID3')
plt.legend()
plt.show()
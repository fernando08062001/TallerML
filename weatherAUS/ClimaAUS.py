import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #libreria del modelo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

simplefilter(action='ignore', category=FutureWarning)

url = 'weatherAUS.csv'
data = pd.read_csv(url)
 
data.RainToday.replace(['No', 'Yes'], [0, 1], inplace= True)
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace= True)

data.drop(['WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'Location', 'MinTemp'
           , 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm'
           , 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm', 'Sunshine','Date'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)

# Partir la data por la mitad (Media pa training y media pa testing)

data_train = data[:71000]
data_test = data[710000:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow) # 0 No llueve hoy 1 si llueve hoy


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow) # 0 No llueve hoy 1 si llueve hoy
#--------------------------------------------------------------------------------------
# REGRESIÓN LOGÍSTICA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)
print(f'Matriz de Y en prediccion: {y_pred}')
print(f'Matriz de y en real :{y_test_out}')

print('*'*50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_1}')
#------------------------------------------------------------------------------------
#random forest

rf = RandomForestClassifier()
for train, test in kfold.split(x, y):
    rf.fit(x[train], y[train])
    scores_train_train = rf.score(x[train], y[train])
    scores_test_train = rf.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)
print(f'Matriz de Y en prediccion: {y_pred}')
print(f'Matriz de y en real :{y_test_out}')

print('*'*50)
print('Random forest con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión de random forest: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución de random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_rforest = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_rforest}')
#----------------------------------------------------------------------------------------
#Maquina de soporte vectorial
svc = SVC(gamma='auto')
for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)
print(f'Matriz de Y en prediccion: {y_pred}')
print(f'Matriz de y en real :{y_test_out}')

print('*'*50)
print('Maquina de soporte vectorial Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión de Maquina de soporte vectorial: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_2 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_2}')

#-----------------------------------------------------------------------------------------------
#KNeighbors

kn = KNeighborsClassifier()
for train, test in kfold.split(x, y):
    kn.fit(x[train], y[train])
    scores_train_train = kn.score(x[train], y[train])
    scores_test_train = kn.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)
print(f'Matriz de Y en prediccion: {y_pred}')
print(f'Matriz de y en real :{y_test_out}')

print('*'*50)
print('KNeighbors con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión de KNeighbors: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución de random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_KN = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_KN}')

#-------------------------------------------------------------------------------------------
#Arbol de desicion
arbol = DecisionTreeClassifier()
for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)
print(f'Matriz de Y en prediccion: {y_pred}')
print(f'Matriz de y en real :{y_test_out}')

print('*'*50)
print('Arbol de desicion con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión de Arbol de desicion: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución de random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_AR = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_AR}')
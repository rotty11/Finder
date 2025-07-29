import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from codecarbon import EmissionsTracker
from sklearn.neural_network import MLPClassifier


def select_features(pheromone_matrix, heuristic, num_features):
    selected_features = []
    remaining_features = list(range(num_features))
    for _ in range(num_features):
        prob = (pheromone_matrix[remaining_features] ** alpha) * (heuristic[remaining_features] ** beta)
        prob /= np.sum(prob)
        selected_feature = np.random.choice(remaining_features, p=prob)
        selected_features.append(selected_feature)
        remaining_features.remove(selected_feature)
    return selected_features


def evaluate_antLGBM(X_train, X_test, y_train, y_test, selected_features):
    knn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    knn.fit(X_train[:, selected_features], y_train)
    y_pred = knn.predict(X_test[:, selected_features])
    return accuracy_score(y_test, y_pred), selected_features


def update_pheromone_matrix(pheromone_matrix, selected_features, scores, rho):
    pheromone_matrix *= (1 - rho)
    for i, score in enumerate(scores):
        for feature in selected_features[i]:
            pheromone_matrix[feature] += score

import numpy as np
import pandas as pd
import scipy.io as sio

orden = "/Users/manuelsanchezjimenez/Documents/TFG/FINAL_1812/FBCSP_datos/09_CSP_ranking_2.csv"
dato_E_CSV = "/Users/manuelsanchezjimenez/Documents/TFG/FINAL_1812/FBCSP_datos/09E_CSP_data.csv"
dato_T_CSV = "/Users/manuelsanchezjimenez/Documents/TFG/FINAL_1812/FBCSP_datos/09T_CSP_data.csv"

nombredataSet = dato_T_CSV
nombredataSetEval = dato_E_CSV

dataset = pd.read_csv(nombredataSet, sep=",", header=None)
x_train = dataset.iloc[:, :].values

datasetTest = pd.read_csv(nombredataSetEval, sep=",", header=None)
X_test = datasetTest.iloc[:, :].values

vectorOrden = pd.read_csv(orden, sep=",", header=None)
vectorOrden = vectorOrden.iloc[:, 0].values

PATH = "/Users/manuelsanchezjimenez/Documents/TFG/PRUEBA 13:11:2023/IV-2a-master/dataset2/"
a = sio.loadmat(PATH + 'A09_training_class.mat')
y_train = a['labels']
b = sio.loadmat(PATH + 'A09_testing_class.mat')
y_test = b['labels']

y_train = y_train[:, 0].ravel()
y_test = y_test[:, 0].ravel()

trainOrdenado = np.take(x_train, vectorOrden, axis=1)
testOrdenado = np.take(X_test,vectorOrden,axis=1)


num_features = 600
num_ants = 12
max_iter = 50
alpha = 1.0
beta = 2.0
rho = 0.5

pheromone_matrix = np.ones(num_features)
best_score = -1
best_features = None

tracker = EmissionsTracker()
tracker.start()
for _ in range(max_iter):
    heuristic = np.random.rand(num_features)
    ant_scores = []
    ant_selected_features = []

    for _ in range(num_ants):
        selected_features = select_features(pheromone_matrix, heuristic, num_features)
        score, selected_features = evaluate_antLGBM(trainOrdenado, testOrdenado, y_train, y_test, selected_features)

        ant_scores.append(score)
        ant_selected_features.append(selected_features)

        if score > best_score:
            best_score = score
            best_features = selected_features

    update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores, rho)

emissions: float = tracker.stop()
print(emissions)

print("Mejores características encontradas:", best_features)
print("Precisión del clasificador LGBM con características seleccionadas:", best_score)

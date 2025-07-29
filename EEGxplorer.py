import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from lightgbm import LGBMClassifier
from pygad import pygad
import random
import math
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from pyswarm import pso
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import warnings
from codecarbon import EmissionsTracker


warnings.filterwarnings("ignore")

orden = "01_CSP_ranking.csv"
dato_E_CSV = "01E_CSP_data.csv"
dato_T_CSV = "01T_CSP_data.csv"

nombredataSet = dato_T_CSV
nombredataSetEval = dato_E_CSV

dataset = pd.read_csv(nombredataSet, sep=",", header=None)
x_train = dataset.iloc[:, :].values

datasetTest = pd.read_csv(nombredataSetEval, sep=",", header=None)
X_test = datasetTest.iloc[:, :].values

vectorOrden = pd.read_csv(orden, sep=",", header=None)
vectorOrden = vectorOrden.iloc[:, 0].values

PATH = "/home/MATLAB/"
a = sio.loadmat(PATH + 'A01_training_class.mat')
y_train = a['labels']
b = sio.loadmat(PATH + 'A01_testing_class.mat')
y_test = b['labels']

y_train = y_train[:, 0].ravel()
y_test = y_test[:, 0].ravel()

trainOrdenado = np.take(x_train, vectorOrden, axis=1)
testOrdenado = np.take(X_test, vectorOrden, axis=1)

alpha = 1.0
beta = 2.0


def select_featuresLGBM(pheromone_matrix, heuristic, num_features):
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
    clf = LGBMClassifier()
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    return accuracy_score(y_test, y_pred), selected_features


def evaluate_antXGBM(X_train, X_test, y_train, y_test, selected_features):
    clf = xgb.XGBClassifier()
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    return accuracy_score(y_test, y_pred), selected_features


def update_pheromone_matrix(pheromone_matrix, selected_features, scores, rho):
    pheromone_matrix *= (1 - rho)
    for i, score in enumerate(scores):
        for feature in selected_features[i]:
            pheromone_matrix[feature] += score


def select_featuresXGBM(pheromone_matrix, heuristic, num_features):
    selected_features = []
    remaining_features = list(range(num_features))
    for _ in range(num_features):
        prob = (pheromone_matrix[remaining_features] ** alpha) * (heuristic[remaining_features] ** beta)
        prob /= np.sum(prob)
        selected_feature = np.random.choice(remaining_features, p=prob)
        selected_features.append(selected_feature)
        remaining_features.remove(selected_feature)
    return selected_features


def clasificador_KNN_GA_Default(NGeneracion, NGenes):
    for i in range(1, 30):
        n_neighbors = i

        # Función de evaluación para el algoritmo genético
        def fitness_func(self, solution, solution_idx):
            # Crear un clasificador KNN con los parámetros dados por la solución
            from sklearn.neighbors import KNeighborsClassifier
            k_value = solution[0]
            knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Entrenar el modelo y predecir las etiquetas
            knn_classifier.fit(trainOrdenado[:, 0:int(k_value)], y_train)
            y_pred = knn_classifier.predict(testOrdenado[:, 0:int(k_value)])

            # Calcular la precisión y devolver como valor de aptitud
            accuracy = accuracy_score(y_test, y_pred)
            fitness = accuracy
            return fitness

        # Definir la configuración del algoritmo genético

        num_generations = NGeneracion
        num_parents_mating = 16
        num_genes = NGenes

        # Rango de valores para el número de vecinos (k) en KNN
        gene_space = [{'low': 1, 'high': 5000}]

        # Crear una instancia del algoritmo genético
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               sol_per_pop=20,
                               num_genes=num_genes,
                               gene_space=gene_space,
                               parent_selection_type="tournament",
                               crossover_probability=0.8,
                               mutation_probability=0.05
                               )

        # Ejecutar el algoritmo genético
        ga_instance.run()

        # Obtener la mejor solución encontrada
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        # Crear el clasificador KNN con la mejor solución
        best_k_value = int(solution[0])

        # Imprimir la mejor solución y su aptitud

        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(trainOrdenado[:, 0:best_k_value], y_train)
        y_pred = knn.predict(testOrdenado[:, 0:best_k_value])
        accuracy = accuracy_score(y_test, y_pred)
        print(
            "Features:" + str(best_k_value) + " - " + " N_NEIGHBORS: " + str(n_neighbors) + " - " + "Accuracy: " + str(
                accuracy))


def clasificador_KNN_GA_NFEATURES(n_neighbors, NGeneracion, NGenes):
    import pygad
    from sklearn.neighbors import KNeighborsClassifier

    def fitness_func(self, solution, solution_idx):
        # Crear un clasificador KNN con los parámetros dados por la solución
        from sklearn.neighbors import KNeighborsClassifier
        k_value = solution[0]
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Entrenar el modelo y predecir las etiquetas
        knn_classifier.fit(trainOrdenado[:, 0:int(k_value)], y_train)
        y_pred = knn_classifier.predict(testOrdenado[:, 0:int(k_value)])

        # Calcular la precisión y devolver como valor de aptitud
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 5000}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    # Imprimir la mejor solución y su aptitud
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(trainOrdenado[:, 0:best_k_value], y_train)
    y_pred = knn.predict(testOrdenado[:, 0:best_k_value])
    accuracy = accuracy_score(y_test, y_pred)
    print("Features:" + str(best_k_value) + " - " + " N_NEIGHBORS: " + str(n_neighbors) + " - " + "Accuracy: " + str(
        accuracy))


def clasificador_KNN_GA_NVECINOS(n_features, NGeneracion, NGenes):
    import pygad
    from sklearn.neighbors import KNeighborsClassifier

    def fitness_func(self, solution, solution_idx):
        # Crear un clasificador KNN con los parámetros dados por la solución
        from sklearn.neighbors import KNeighborsClassifier
        k_value = int(solution[0])
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

        # Entrenar el modelo y predecir las etiquetas
        knn_classifier.fit(trainOrdenado[:, 0:int(n_features)], y_train)
        y_pred = knn_classifier.predict(testOrdenado[:, 0:int(n_features)])

        # Calcular la precisión y devolver como valor de aptitud
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 40}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    # Imprimir la mejor solución y su aptitud
    knn = KNeighborsClassifier(n_neighbors=best_k_value)
    knn.fit(trainOrdenado[:, 0:n_features], y_train)
    y_pred = knn.predict(testOrdenado[:, 0:n_features])
    accuracy = accuracy_score(y_test, y_pred)
    print("Features:" + str(n_features) + " - " + " N_NEIGHBORS: " + str(best_k_value) + " - " + "Accuracy: " + str(
        accuracy))


def clasificador_SVC_GA_Default(C, NGeneracion, NGenes):
    def fitness_func(self, solution, solution_idx):
        from sklearn.svm import LinearSVC

        k_value = solution[0]

        clf = LinearSVC(C=C, intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr', penalty='l2',
                        random_state=1, tol=0.00001)
        clf.fit(trainOrdenado[:, 0:int(k_value)], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:int(k_value)])
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 4000}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=50,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    # Imprimir la mejor solución y su aptitud
    print("Número óptimo de features (N):", best_k_value)
    clf = LinearSVC(C=C, intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr', penalty='l2',
                    random_state=1, tol=0.00001)
    clf.fit(trainOrdenado[:, 0:int(best_k_value)], y_train)
    y_pred = clf.predict(testOrdenado[:, 0:int(best_k_value)])
    accuracy = accuracy_score(y_test, y_pred)

    print("Número óptimo de features (N):", str(best_k_value) + " ACCURACY: " + str(accuracy))


def clasificador_MLP_GA(NGeneracion, NGenes):
    def fitness_func(self, solution, solution_idx):
        from sklearn.svm import LinearSVC

        k_value = solution[0]

        clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant',
                            solver='sgd')
        clf.fit(trainOrdenado[:, 0:int(k_value)], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:int(k_value)])
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 11352}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    # Imprimir la mejor solución y su aptitud
    print("Número óptimo de features (k):", best_k_value)
    clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant',
                        solver='sgd')
    clf.fit(trainOrdenado[:, 0:int(best_k_value)], y_train)
    y_pred = clf.predict(testOrdenado[:, 0:int(best_k_value)])
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)


def clasificador_XGBM_GA(NGeneracion, NGenes):
    def fitness_func(self, solution, solution_idx):
        k_value = solution[0]

        clf = xgb.XGBClassifier()
        clf.fit(trainOrdenado[:, 0:int(k_value)], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:int(k_value)])
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 11352}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    print("Número óptimo de features (k):", best_k_value)
    clf = xgb.XGBClassifier()
    clf.fit(trainOrdenado[:, 0:int(best_k_value)], y_train)
    y_pred = clf.predict(testOrdenado[:, 0:int(best_k_value)])
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)


def clasificador_LGBM_GA(NGeneracion, NGenes):
    def fitness_func(self, solution, solution_idx):
        k_value = solution[0]

        clf = LGBMClassifier()
        clf.fit(trainOrdenado[:, 0:int(k_value)], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:int(k_value)])
        accuracy = accuracy_score(y_test, y_pred)
        fitness = accuracy
        return fitness

    # Definir la configuración del algoritmo genético

    num_generations = NGeneracion
    num_parents_mating = 16
    num_genes = NGenes

    # Rango de valores para el número de vecinos (k) en KNN
    gene_space = [{'low': 1, 'high': 11352}]

    # Crear una instancia del algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="tournament",
                           crossover_probability=0.8,
                           mutation_probability=0.05
                           )

    # Ejecutar el algoritmo genético
    ga_instance.run()

    # Obtener la mejor solución encontrada
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Crear el clasificador KNN con la mejor solución
    best_k_value = int(solution[0])

    print("Número óptimo de features (k):", best_k_value)
    clf = LGBMClassifier()
    clf.fit(trainOrdenado[:, 0:int(best_k_value)], y_train)
    y_pred = clf.predict(testOrdenado[:, 0:int(best_k_value)])
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)


def clasificador_KNN_PSO_Default():
    for i in range(1, 30):
        n_neighbors = i

        def evaluate_knn(params):
            k = int(params[0])  # Valor de k para KNN
            knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_classifier.fit(trainOrdenado[:, 0:k], y_train)
            y_pred = knn_classifier.predict(testOrdenado[:, 0:k])
            accuracy = accuracy_score(y_test, y_pred)
            return -accuracy  # Se utiliza el negativo de la precisión para la maximización

        # Definición de límites para los parámetros
        lb = [1]  # Límite inferior de k
        ub = [11352]  # Límite superior de k

        # Aplicar PSO para optimizar el valor de k
        k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

        # Entrenar KNN con el valor óptimo de k
        k_opt = int(k_opt[0])
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier.fit(trainOrdenado[:, 0:k_opt], y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred = knn_classifier.predict(testOrdenado[:, 0:k_opt])

        # Calcular la precisión del clasificador
        accuracy = accuracy_score(y_test, y_pred)
        print("Features:" + str(k_opt) + " - " + " N_NEIGHBORS: " + str(n_neighbors) + " - " + "Accuracy: " + str(
            accuracy))


def clasificador_KNN_PSO_NFEATURES(n_neighbors):
    def evaluate_knn(params):
        k = int(params[0])  # Valor de k para KNN
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier.fit(trainOrdenado[:, 0:k], y_train)
        y_pred = knn_classifier.predict(testOrdenado[:, 0:k])
        accuracy = accuracy_score(y_test, y_pred)
        return -accuracy  # Se utiliza el negativo de la precisión para la maximización

    # Definición de límites para los parámetros
    lb = [1]  # Límite inferior de k
    ub = [11352]  # Límite superior de k

    # Aplicar PSO para optimizar el valor de k
    k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

    # Entrenar KNN con el valor óptimo de k
    k_opt = int(k_opt[0])
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(trainOrdenado[:, 0:k_opt], y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = knn_classifier.predict(testOrdenado[:, 0:k_opt])

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print("Features:" + str(k_opt) + " - " + " N_NEIGHBORS: " + str(n_neighbors) + " - " + "Accuracy: " + str(
        accuracy))


def clasificador_KNN_PSO_NVECINOS(n_features):
    def evaluate_knn(params):
        k = int(params[0])  # Valor de k para KNN
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(trainOrdenado[:, 0:n_features], y_train)
        y_pred = knn_classifier.predict(testOrdenado[:, 0:n_features])
        accuracy = accuracy_score(y_test, y_pred)
        return -accuracy  # Se utiliza el negativo de la precisión para la maximización

    # Definición de límites para los parámetros
    lb = [1]  # Límite inferior de k
    ub = [30]  # Límite superior de k

    # Aplicar PSO para optimizar el valor de k
    k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

    # Entrenar KNN con el valor óptimo de k
    k_opt = int(k_opt[0])
    knn_classifier = KNeighborsClassifier(n_neighbors=k_opt)
    knn_classifier.fit(trainOrdenado[:, 0:n_features], y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = knn_classifier.predict(testOrdenado[:, 0:n_features])

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print("Features:" + str(n_features) + " - " + " N_NEIGHBORS: " + str(k_opt) + " - " + "Accuracy: " + str(
        accuracy))


def clasificador_SVC_PSO_Default(C):
    def evaluate_knn(params):
        k = int(params[0])
        SVC_classifier = SVC(C=C, gamma='scale')
        SVC_classifier.fit(trainOrdenado[:, 0:k], y_train)
        y_pred = SVC_classifier.predict(testOrdenado[:, 0:k])
        accuracy = accuracy_score(y_test, y_pred)
        return -accuracy  # Se utiliza el negativo de la precisión para la maximización

    # Definición de límites para los parámetros
    lb = [1]  # Límite inferior de k
    ub = [11352]  # Límite superior de k

    # Aplicar PSO para optimizar el valor de k
    k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

    # Entrenar KNN con el valor óptimo de k
    k_opt = int(k_opt[0])
    knn_classifier = SVC(C=C, gamma='scale')
    knn_classifier.fit(trainOrdenado[:, 0:k_opt], y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = knn_classifier.predict(testOrdenado[:, 0:k_opt])

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejor valor de N features: {k_opt}")
    print("Accuracy:", accuracy)


def clasificador_XGBM_PSO_Default():
    def evaluate_knn(params):
        k = int(params[0])
        clf = xgb.XGBClassifier()
        clf.fit(trainOrdenado[:, 0:k], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:k])
        accuracy = accuracy_score(y_test, y_pred)
        return -accuracy  # Se utiliza el negativo de la precisión para la maximización

    # Definición de límites para los parámetros
    lb = [1]  # Límite inferior de k
    ub = [11352]  # Límite superior de k

    # Aplicar PSO para optimizar el valor de k
    k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

    # Entrenar KNN con el valor óptimo de k
    k_opt = int(k_opt[0])
    clf = xgb.XGBClassifier()
    clf.fit(trainOrdenado[:, 0:k_opt], y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = clf.predict(testOrdenado[:, 0:k_opt])

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejor valor de N features: {k_opt}")
    print("Accuracy:", accuracy)


def clasificador_LGBM_PSO_Default():
    def evaluate_knn(params):
        k = int(params[0])
        clf = LGBMClassifier()
        clf.fit(trainOrdenado[:, 0:k], y_train)
        y_pred = clf.predict(testOrdenado[:, 0:k])
        accuracy = accuracy_score(y_test, y_pred)
        return -accuracy  # Se utiliza el negativo de la precisión para la maximización

    # Definición de límites para los parámetros
    lb = [1]  # Límite inferior de k
    ub = [11352]  # Límite superior de k

    # Aplicar PSO para optimizar el valor de k
    k_opt, _ = pso(evaluate_knn, lb, ub, swarmsize=10, maxiter=20)

    # Entrenar KNN con el valor óptimo de k
    k_opt = int(k_opt[0])
    clf = LGBMClassifier()
    clf.fit(trainOrdenado[:, 0:k_opt], y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = clf.predict(testOrdenado[:, 0:k_opt])

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejor valor de N features: {k_opt}")
    print("Accuracy:", accuracy)


def clasificador_MLP_PSO():
    def optimize_mlp(params):
        k_value = int(params[0])

        # Creamos el clasificador MLP con los parámetros proporcionados
        mlp = MLPClassifier()

        # Entrenamos el clasificador
        mlp.fit(x_train[:,0:k_value], y_train)

        # Realizamos predicciones en el conjunto de prueba
        y_pred = mlp.predict(X_test[:,0:k_value])

        # Calculamos la precisión como la métrica objetivo a maximizar
        accuracy = accuracy_score(y_test, y_pred)

        # Devolvemos el negativo de la precisión porque PSO minimiza la función objetivo
        return -accuracy

    # Definimos los límites para los hiperparámetros
    lb = [1]  # Límites inferiores para el tamaño de la capa oculta, alpha y tasa de aprendizaje
    ub = [11352]  # Límites superiores para el tamaño de la capa oculta, alpha y tasa de aprendizaje

    # Ejecutamos PSO para optimizar los hiperparámetros
    best_params, _ = pso(optimize_mlp, lb, ub)

    # Creamos el clasificador MLP final con los mejores hiperparámetros encontrados
    k_value = (int(best_params[0]))

    final_mlp = MLPClassifier()

    # Entrenamos el clasificador final
    final_mlp.fit(x_train[:,0:k_value], y_train)

    # Realizamos predicciones en el conjunto de prueba
    y_pred = final_mlp.predict(X_test[:,0:k_value])

    # Calculamos la precisión final
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def clasificador_KNN_ES_Default():
    for i in range(1, 40):
        n_neighbors = i

        # Función para generar una solución vecina
        def generate_neighbor(solution, max_k):
            n_neighbors = random.randint(1, 6000)
            return n_neighbors

        # Función de enfriamiento simulado
        def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                                max_iterations):
            current_k = initial_k
            current_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            current_classifier.fit(X_train[:, 0:int(current_k)], y_train)
            current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:int(current_k)]))

            best_k = current_k
            best_accuracy = current_accuracy

            for iteration in range(max_iterations):
                new_k = generate_neighbor(current_k, max_k)
                new_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                new_classifier.fit(X_train[:, 0:int(new_k)], y_train)
                new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:int(new_k)]))

                if new_accuracy > current_accuracy or random.random() < math.exp(
                        (new_accuracy - current_accuracy) / temperature):
                    current_k = new_k
                    current_classifier = new_classifier
                    current_accuracy = new_accuracy

                if current_accuracy > best_accuracy:
                    best_k = current_k
                    best_accuracy = current_accuracy

                temperature *= cooling_rate

            return best_k, best_accuracy

        # Ejemplo de uso
        # Supongamos que tienes tus datos en X e Y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Parámetros de enfriamiento simulado
        initial_k = 2
        max_k = 11352
        temperature = 1.0
        cooling_rate = 0.95
        max_iterations = 2000

        trainOrdenado = np.take(x_train, vectorOrden, axis=1)
        testOrdenado = np.take(X_test, vectorOrden, axis=1)

        best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                    temperature, cooling_rate, max_iterations)

        print("Features:" + str(best_k) + " - " + " N_NEIGHBORS: " + str(n_neighbors) + " - " + "Accuracy: " + str(
            best_accuracy))


def clasificador_KNN_ES_NFEATURES(n_neighbors):
    def generate_neighbor(solution, max_k):
        n_neighbors = random.randint(1, 4000)
        return n_neighbors

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        current_classifier.fit(X_train[:, 0:int(current_k)], y_train)
        current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:int(current_k)]))

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)
            new_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            new_classifier.fit(X_train[:, 0:int(new_k)], y_train)
            new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:int(new_k)]))

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = 11352
    temperature = 1.0
    cooling_rate = 0.95
    max_iterations = 2000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Mejor valor de k encontrado: {best_k}")
    print(f"Exactitud correspondiente: {best_accuracy}")
    print(f"n_neighbors: {n_neighbors}")


def clasificador_KNN_ES_NVECINOS(n_features):
    def generate_neighbor(solution, max_k):
        n_neighbors = random.randint(1, 40)
        return n_neighbors

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = KNeighborsClassifier(n_neighbors=current_k)
        current_classifier.fit(X_train[:, 0:n_features], y_train)
        current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:n_features]))

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)
            new_classifier = KNeighborsClassifier(n_neighbors=new_k)
            new_classifier.fit(X_train[:, 0:n_features], y_train)
            new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:n_features]))

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = 40
    temperature = 1.0
    cooling_rate = 0.98
    max_iterations = 1000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Features: {n_features}")
    print(f"Exactitud correspondiente: {best_accuracy}")
    print(f"n_neighbors: {best_k}")


def clasificador_LGBM_ES(n_features):
    def generate_neighbor(solution, max_k):
        neighbor = random.randint(1, n_features)
        return neighbor

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = LGBMClassifier()
        current_classifier.fit(X_train[:, 0:current_k], y_train)
        y_pred = current_classifier.predict(X_test[:, 0:current_k])
        current_accuracy = accuracy_score(y_val, y_pred)

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)

            new_classifier = LGBMClassifier()
            new_classifier.fit(X_train[:, 0:new_k], y_train)
            y_pred_test = new_classifier.predict(X_val[:, 0:new_k])
            new_accuracy = accuracy_score(y_val, y_pred_test)

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = n_features
    temperature = 1.0
    cooling_rate = 0.98
    max_iterations = 1000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Mejor valor de k encontrado: {best_k}")
    print(f"Exactitud correspondiente: {best_accuracy}")


def clasificador_XGBM_ES(n_features):
    def generate_neighbor(solution, max_k):
        neighbor = random.randint(1, n_features)
        return neighbor

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = xgb.XGBClassifier()
        current_classifier.fit(X_train[:, 0:current_k], y_train)
        current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:current_k]))

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)
            new_classifier = xgb.XGBClassifier()
            new_classifier.fit(X_train[:, 0:new_k], y_train)
            new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:new_k]))

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = n_features
    temperature = 1.0
    cooling_rate = 0.98
    max_iterations = 1000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Mejor valor de k encontrado: {best_k}")
    print(f"Exactitud correspondiente: {best_accuracy}")

def clasificador_MLP_ES(NFeatures):
    def generate_neighbor(solution, max_k):
        neighbor = random.randint(1, NFeatures)
        return neighbor

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = MLPClassifier()
        current_classifier.fit(X_train[:, 0:current_k], y_train)
        current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:current_k]))

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)
            new_classifier = MLPClassifier()
            new_classifier.fit(X_train[:, 0:new_k], y_train)
            new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:new_k]))

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = NFeatures
    temperature = 1.0
    cooling_rate = 0.98
    max_iterations = 1000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Mejor valor de k encontrado: {best_k}")
    print(f"Exactitud correspondiente: {best_accuracy}")

def clasificador_SVC_ES_Default(C,n_features):
    def generate_neighbor(solution, max_k):
        n_neighbors = random.randint(1, n_features)
        return n_neighbors

    # Función de enfriamiento simulado
    def simulated_annealing(X_train, y_train, X_val, y_val, initial_k, max_k, temperature, cooling_rate,
                            max_iterations):
        current_k = initial_k
        current_classifier = SVC(C=C, gamma='scale')
        current_classifier.fit(X_train[:, 0:current_k], y_train)
        current_accuracy = accuracy_score(y_val, current_classifier.predict(X_val[:, 0:current_k]))

        best_k = current_k
        best_accuracy = current_accuracy

        for iteration in range(max_iterations):
            new_k = generate_neighbor(current_k, max_k)
            new_classifier = SVC(C=C, gamma='scale')
            new_classifier.fit(X_train[:, 0:new_k], y_train)
            new_accuracy = accuracy_score(y_val, new_classifier.predict(X_val[:, 0:new_k]))

            if new_accuracy > current_accuracy or random.random() < math.exp(
                    (new_accuracy - current_accuracy) / temperature):
                current_k = new_k
                current_classifier = new_classifier
                current_accuracy = new_accuracy

            if current_accuracy > best_accuracy:
                best_k = current_k
                best_accuracy = current_accuracy

            temperature *= cooling_rate

        return best_k, best_accuracy

    # Ejemplo de uso
    # Supongamos que tienes tus datos en X y y, asegúrate de dividirlos en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parámetros de enfriamiento simulado
    initial_k = 2
    max_k = n_features
    temperature = 1.0
    cooling_rate = 0.98
    max_iterations = 1000

    trainOrdenado = np.take(x_train, vectorOrden, axis=1)
    testOrdenado = np.take(X_test, vectorOrden, axis=1)

    best_k, best_accuracy = simulated_annealing(trainOrdenado, y_train, testOrdenado, y_test, initial_k, max_k,
                                                temperature, cooling_rate, max_iterations)

    print(f"Mejor valor de N features: {best_k}")
    print(f"Accuracy: {best_accuracy}")


def clasificador_KNN_HORMIGAS(Nvecinos, Nfeatures, Nhormigas):
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

    def evaluate_ant(X_train, X_test, y_train, y_test, selected_features):
        knn = KNeighborsClassifier(n_neighbors=Nvecinos)
        knn.fit(X_train[:, selected_features], y_train)
        y_pred = knn.predict(X_test[:, selected_features])
        return accuracy_score(y_test, y_pred), selected_features

    def update_pheromone_matrix(pheromone_matrix, selected_features, scores, rho):
        pheromone_matrix *= (1 - rho)
        for i, score in enumerate(scores):
            for feature in selected_features[i]:
                pheromone_matrix[feature] += score

    num_features = Nfeatures
    num_ants = Nhormigas
    max_iter = 50
    alpha = 1.0
    beta = 2.0
    rho = 0.5

    pheromone_matrix = np.ones(num_features)
    best_score = -1
    best_features = None

    for _ in range(max_iter):
        heuristic = np.random.rand(num_features)
        ant_scores = []
        ant_selected_features = []

        for _ in range(num_ants):
            selected_features = select_features(pheromone_matrix, heuristic, num_features)
            score, selected_features = evaluate_ant(trainOrdenado, testOrdenado, y_train, y_test, selected_features)

            ant_scores.append(score)
            ant_selected_features.append(selected_features)

            if score > best_score:
                best_score = score
                best_features = selected_features

        update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores, rho)

    print("Precisión del clasificador k-NN con N-" + str(Nvecinos) + " NFeatures- " + str(Nfeatures) + " : ",
          best_score)


def clasificador_SVC_HORMIGAS(C):
    def evaluate_svm(X_train, X_test, y_train, y_test, params):
        C = params[0]
        # gamma = params[1]
        svm = SVC(C=C, gamma='scale', kernel='rbf')
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características basada en la matriz de feromonas y heurística
    def select_features(pheromone_matrix, heuristic, num_features):
        selected_features = []
        remaining_features = list(range(num_features))
        for _ in range(num_features):
            prob = pheromone_matrix[remaining_features] * heuristic[remaining_features]
            prob /= np.sum(prob)
            selected_feature = np.random.choice(remaining_features, p=prob)
            selected_features.append(selected_feature)
            remaining_features.remove(selected_feature)
        return selected_features

    # Función para actualizar la matriz de feromonas
    def update_pheromone_matrix(pheromone_matrix, selected_features, scores):
        pheromone_matrix *= 0.5  # Factor de evaporación
        for i, score in enumerate(scores):
            for feature in selected_features[i]:
                pheromone_matrix[feature] += score

    # Algoritmo de colonia de hormigas para optimizar SVM
    def ant_colony_optimization(X_train, X_test, y_train, y_test, C, num_ants=10, max_iter=100, num_features=None):
        if num_features is None:
            num_features = X_train.shape[1]
        pheromone_matrix = np.ones(num_features)
        best_score = -1
        best_features = None

        for _ in range(max_iter):
            heuristic = np.random.rand(num_features)
            ant_scores = []
            ant_selected_features = []

            for _ in range(num_ants):
                selected_features = select_features(pheromone_matrix, heuristic, num_features)
                score = evaluate_svm(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test,
                                     [C, 'scale'])
                ant_scores.append(score)
                ant_selected_features.append(selected_features)

                if score > best_score:
                    best_score = score
                    best_features = selected_features

            update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores)

        return best_features, best_score

    best_features, best_score = ant_colony_optimization(x_train, X_test, y_train, y_test, C)
    print("Precisión del clasificador SVM con características seleccionadas:", best_score)


def clasificador_XGBM_HORMIGAS(Nfeatures, NHormigas):
    num_features = Nfeatures
    num_ants = NHormigas
    max_iter = 50
    rho = 0.5
    pheromone_matrix = np.ones(num_features)
    best_score = -1
    best_features = None

    for _ in range(max_iter):
        heuristic = np.random.rand(num_features)
        ant_scores = []
        ant_selected_features = []

        for _ in range(num_ants):
            selected_features = select_featuresXGBM(pheromone_matrix, heuristic, num_features)
            score, selected_features = evaluate_antXGBM(trainOrdenado, testOrdenado, y_train, y_test, selected_features)

            ant_scores.append(score)
            ant_selected_features.append(selected_features)

            if score > best_score:
                best_score = score
                best_features = selected_features

        update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores, rho)

    print("Precisión del clasificador XGBM con características seleccionadas:", best_score)


def clasificador_LGBM_HORMIGAS(NFeatures, NHormigas):
    num_features = NFeatures
    num_ants = NHormigas
    max_iter = 50
    rho = 0.5

    pheromone_matrix = np.ones(num_features)
    best_score = -1
    best_features = None

    for _ in range(max_iter):
        heuristic = np.random.rand(num_features)
        ant_scores = []
        ant_selected_features = []

        for _ in range(num_ants):
            selected_features = select_featuresLGBM(pheromone_matrix, heuristic, num_features)
            score, selected_features = evaluate_antLGBM(trainOrdenado, testOrdenado, y_train, y_test, selected_features)

            ant_scores.append(score)
            ant_selected_features.append(selected_features)

            if score > best_score:
                best_score = score
                best_features = selected_features

        update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores, rho)

    print("Precisión del clasificador LGBM con características seleccionadas:", best_score)


def clasificador_MLP_HORMIGAS(Nfeatures, Nhormigas):
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

    def evaluate_ant(X_train, X_test, y_train, y_test, selected_features):
        knn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        knn.fit(X_train[:, selected_features], y_train)
        y_pred = knn.predict(X_test[:, selected_features])
        return accuracy_score(y_test, y_pred), selected_features

    def update_pheromone_matrix(pheromone_matrix, selected_features, scores, rho):
        pheromone_matrix *= (1 - rho)
        for i, score in enumerate(scores):
            for feature in selected_features[i]:
                pheromone_matrix[feature] += score

    num_features = Nfeatures
    num_ants = Nhormigas
    max_iter = 50
    alpha = 1.0
    beta = 2.0
    rho = 0.5

    pheromone_matrix = np.ones(num_features)
    best_score = -1
    best_features = None

    for _ in range(max_iter):
        heuristic = np.random.rand(num_features)
        ant_scores = []
        ant_selected_features = []

        for _ in range(num_ants):
            selected_features = select_features(pheromone_matrix, heuristic, num_features)
            score, selected_features = evaluate_ant(trainOrdenado, testOrdenado, y_train, y_test, selected_features)

            ant_scores.append(score)
            ant_selected_features.append(selected_features)

            if score > best_score:
                best_score = score
                best_features = selected_features

        update_pheromone_matrix(pheromone_matrix, ant_selected_features, ant_scores, rho)

    print("Precisión del clasificador MLP con " + str(Nfeatures) + " Features y características seleccionadas:",
          best_score)


def clasificador_KNN_BEE(Nvecinos, Nfeatures, ):
    def evaluate_model(X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(n_neighbors=Nvecinos)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características utilizando BCO
    def bee_colony_optimization(X_train, X_test, y_train, y_test, num_iterations, num_selected_features):
        num_features = X_train.shape[1]
        best_feature_set = None
        best_accuracy = 0

        for _ in range(num_iterations):
            # Generar una solución aleatoria
            selected_features = np.random.choice(range(num_features), num_selected_features, replace=False)

            # Evaluar el rendimiento del modelo utilizando las características seleccionadas
            accuracy = evaluate_model(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test)

            # Actualizar el mejor conjunto de características si es necesario
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = selected_features

        return best_feature_set

    # Parámetros para BCO
    num_iterations = 100
    num_selected_features = Nfeatures  # Número de características a seleccionar
    # Ejecutar BCO para seleccionar características
    selected_features = bee_colony_optimization(x_train, X_test, y_train, y_test, num_iterations,
                                                num_selected_features)

    print("Precisión utilizando " + str(Nfeatures) + "Features y " + str(Nvecinos) + " vecinos:",
          evaluate_model(x_train[:, selected_features], X_test[:, selected_features], y_train, y_test))


def clasificador_SVC_BEE(C, Nfeatures):
    def evaluate_model(X_train, X_test, y_train, y_test):
        model = SVC(C=C, gamma='scale', kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características utilizando BCO
    def bee_colony_optimization(X_train, X_test, y_train, y_test, num_iterations, num_selected_features):
        num_features = X_train.shape[1]
        best_feature_set = None
        best_accuracy = 0

        for _ in range(num_iterations):
            # Generar una solución aleatoria
            selected_features = np.random.choice(range(num_features), num_selected_features, replace=False)

            # Evaluar el rendimiento del modelo utilizando las características seleccionadas
            accuracy = evaluate_model(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test)

            # Actualizar el mejor conjunto de características si es necesario
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = selected_features

        return best_feature_set

    # Parámetros para BCO
    num_iterations = 100
    num_selected_features = Nfeatures  # Número de características a seleccionar
    # Ejecutar BCO para seleccionar características
    selected_features = bee_colony_optimization(x_train, X_test, y_train, y_test, num_iterations,
                                                num_selected_features)

    print("Precisión utilizando " + str(Nfeatures) + "Features y " + str(C) + " C:",
          evaluate_model(x_train[:, selected_features], X_test[:, selected_features], y_train, y_test))


def clasificador_XGBM_BEE(Nfeatures):
    def evaluate_model(X_train, X_test, y_train, y_test):
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características utilizando BCO
    def bee_colony_optimization(X_train, X_test, y_train, y_test, num_iterations, num_selected_features):
        num_features = X_train.shape[1]
        best_feature_set = None
        best_accuracy = 0

        for _ in range(num_iterations):
            # Generar una solución aleatoria
            selected_features = np.random.choice(range(num_features), num_selected_features, replace=False)

            # Evaluar el rendimiento del modelo utilizando las características seleccionadas
            accuracy = evaluate_model(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test)

            # Actualizar el mejor conjunto de características si es necesario
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = selected_features

        return best_feature_set

    num_iterations = 100
    num_selected_features = Nfeatures  # Número de características a seleccionar

    # Ejecutar BCO para seleccionar características
    selected_features = bee_colony_optimization(x_train, X_test, y_train, y_test, num_iterations, num_selected_features)
    print("Precisión utilizando el mejor conjunto de características:",
          evaluate_model(x_train[:, selected_features], X_test[:, selected_features], y_train, y_test))

def clasificador_LGBM_BEE(Nfeatures):
    def evaluate_model(X_train, X_test, y_train, y_test):
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características utilizando BCO
    def bee_colony_optimization(X_train, X_test, y_train, y_test, num_iterations, num_selected_features):
        num_features = X_train.shape[1]
        best_feature_set = None
        best_accuracy = 0

        for _ in range(num_iterations):
            # Generar una solución aleatoria
            selected_features = np.random.choice(range(num_features), num_selected_features, replace=False)

            # Evaluar el rendimiento del modelo utilizando las características seleccionadas
            accuracy = evaluate_model(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test)

            # Actualizar el mejor conjunto de características si es necesario
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = selected_features

        return best_feature_set

    num_iterations = 100
    num_selected_features = Nfeatures  # Número de características a seleccionar

    # Ejecutar BCO para seleccionar características
    selected_features = bee_colony_optimization(x_train, X_test, y_train, y_test, num_iterations, num_selected_features)
    print("Precisión utilizando el mejor conjunto de características:",
          evaluate_model(x_train[:, selected_features], X_test[:, selected_features], y_train, y_test))

def clasificador_MLP_BEE(Nfeatures):
    def evaluate_model(X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        model.fit(X_train[:, 0:Nfeatures], y_train)
        y_pred = model.predict(X_test[:, 0:Nfeatures])
        return accuracy_score(y_test, y_pred)

    # Función para seleccionar características utilizando BCO
    def bee_colony_optimization(X_train, X_test, y_train, y_test, num_iterations, num_selected_features):
        num_features = X_train.shape[1]
        best_feature_set = None
        best_accuracy = 0

        for _ in range(num_iterations):
            # Generar una solución aleatoria
            selected_features = np.random.choice(range(num_features), num_selected_features, replace=False)

            # Evaluar el rendimiento del modelo utilizando las características seleccionadas
            accuracy = evaluate_model(X_train[:, selected_features], X_test[:, selected_features], y_train, y_test)

            # Actualizar el mejor conjunto de características si es necesario
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = selected_features

        return best_feature_set

    num_iterations = 100
    num_selected_features = Nfeatures  # Número de características a seleccionar

    # Ejecutar BCO para seleccionar características
    selected_features = bee_colony_optimization(x_train, X_test, y_train, y_test, num_iterations,
                                                num_selected_features)
    print("Precisión utilizando el mejor conjunto de características:",
          evaluate_model(x_train[:, selected_features], X_test[:, selected_features], y_train, y_test))


def main(args):
    if args.MW == "GA":
        print("Algoritmos Geneticos")
        if args.clasificador == 'KNN' and args.vecinos == 0 and args.features == 0:
            print("Clasificador KNN por defecto")
            proyecto = "GA_KNN_defecto";
            #tracker = EmissionsTracker(project_name=proyecto)
            #tracker.start()
            clasificador_KNN_GA_Default(args.NGeneracion, args.NGenes)
            #tracker.stop()
            exit
        elif args.clasificador == 'KNN' and args.vecinos != 0:
            print("Clasificador KNN con K=" + str(args.vecinos))
            proyecto = "GA_KNN_K" + str(args.vecinos);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_GA_NFEATURES(args.vecinos, args.NGeneracion, args.NGenes)
            tracker.stop()
            exit
        elif args.clasificador == 'KNN' and args.features != 0:
            print("Clasificador KNN con N features =" + str(args.features))
            proyecto = "GA_KNN_N" + str(args.features) + "_NGeneracion" + str(args.NGeneracion) + "_NGenes" + str(
                args.NGenes);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_GA_NVECINOS(args.features, args.NGeneracion, args.NGenes)
            tracker.stop()
            exit
        elif args.clasificador == 'SVC' and args.C != 0.1:
            print("Clasificador SVC con C=" + str(args.C))
            proyecto = "GA_SVC_C" + str(args.C);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_GA_Default(args.C, args.NGeneracion, args.NGenes)
            tracker.stop()
            exit
        elif args.clasificador == 'SVC' and args.C == 0.1:
            print("Clasificador SVC con C=0.1")
            proyecto = "GA_SVC_C" + str(args.C) + "_NGeneracion" + str(args.NGeneracion) + "_NGenes" + str(
                args.NGenes);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_GA_Default(args.C, args.NGeneracion, args.NGenes)
            tracker.stop()
        elif args.clasificador == 'MLP':
            print(
                "Clasificador MLP con Num Generaciones " + str(args.NGeneracion) + " y Num Genes: " + str(args.NGenes))
            proyecto = "GA_MLP_NGeneracion" + str(args.NGeneracion) + "_NGenes" + str(args.NGenes);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_MLP_GA(args.NGeneracion, args.NGenes)
            tracker.stop()
        elif args.clasificador == 'XGBM':
            print(
                "Clasificador XGBM con Num Generaciones " + str(args.NGeneracion) + " y Num Genes: " + str(args.NGenes))
            proyecto = "GA_XGBM_NGeneracion" + str(args.NGeneracion) + "_NGenes" + str(args.NGenes);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_XGBM_GA(args.NGeneracion, args.NGenes)
            tracker.stop()
        elif args.clasificador == 'LGBM':
            print(
                "Clasificador LGBM con Num Generaciones " + str(args.NGeneracion) + " y Num Genes: " + str(args.NGenes))
            proyecto = "GA_LGBM_NGeneracion" + str(args.NGeneracion) + "_NGenes" + str(args.NGenes);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_LGBM_GA(args.NGeneracion, args.NGenes)
            tracker.stop()

    if args.MW == "PSO":
        print("Algoritmos PSO")
        if args.clasificador == 'KNN' and args.vecinos == 0 and args.features == 0:
            print("Clasificador KNN por defecto")
            proyecto = "PSO_KNN_defecto";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_PSO_Default()
            tracker.stop()
        elif args.clasificador == 'KNN' and args.vecinos != 0:
            print("Clasificador KNN con K=" + str(args.vecinos))
            proyecto = "PSO_KNN_K" + str(args.vecinos);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_PSO_NFEATURES(args.vecinos)
            tracker.stop()
        elif args.clasificador == 'KNN' and args.features != 0:
            print("Clasificador KNN con N features =" + str(args.features))
            proyecto = "PSO_KNN_N" + str(args.features);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_PSO_NVECINOS(args.features)
            tracker.stop()
        elif args.clasificador == 'SVC' and args.C != 0.1:
            print("Clasificador SVC con C=" + str(args.C))
            proyecto = "PSO_SVC_C" + str(args.C);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_PSO_Default(args.C)
            tracker.stop()
        elif args.clasificador == 'SVC' and args.C == 0.1:
            print("Clasificador SVC con C=0.1")
            proyecto = "PSO_SVC_C" + str(args.C);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_PSO_Default(args.C)
            tracker.stop()
        elif args.clasificador == 'MLP':
            print("Clasificador MLP")
            proyecto = "PSO_MLP";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_MLP_PSO()
            tracker.stop()
        elif args.clasificador == 'XGBM':
            print("Clasificador XGBM")
            proyecto = "PSO_XGBM";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_XGBM_PSO_Default()
            tracker.stop()
        elif args.clasificador == 'LGBM':
            print("Clasificador LGBM")
            proyecto = "PSO_LGBM" + str(args.features);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_LGBM_PSO_Default()
            tracker.stop()
    if args.MW == "ES":
        print("Algoritmos ES")
        if args.clasificador == 'KNN' and args.vecinos == 0 and args.features == 0:
            print("Clasificador KNN por defecto")
            proyecto = "ES_KNN_defecto";
            #tracker = EmissionsTracker(project_name=proyecto)
            #tracker.start()
            clasificador_KNN_ES_Default()
            #tracker.stop()
        elif args.clasificador == 'KNN' and args.vecinos != 0:
            print("Clasificador KNN con K=" + str(args.vecinos))
            proyecto = "ES_KNN_K" + str(args.vecinos)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_ES_NFEATURES(args.vecinos)
            tracker.stop()
        elif args.clasificador == 'KNN' and args.features != 0:
            print("Clasificador KNN con N features =" + str(args.features))
            proyecto = "ES_KNN_N" + str(args.features)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_ES_NVECINOS(args.features)
            tracker.stop()
        elif args.clasificador == 'SVC' and args.C != 1.0:
            print("Clasificador SVC con C=" + str(args.C))
            proyecto = "ES_SVC_C" + str(args.features)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_ES_Default(args.C,args.features)
            tracker.stop()
        elif args.clasificador == 'SVC' and args.C == 1.0:
            print("Clasificador SVC con C=1.0 (Default)")
            proyecto = "ES_SVC_C" + str(args.features)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_ES_Default(args.C,args.features)
            tracker.stop()
        elif args.clasificador == 'XGBM':
            print("Clasificador XGBM")
            proyecto = "ES_XGBM"
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_XGBM_ES(args.features)
            tracker.stop()
        elif args.clasificador == 'LGBM':
            print("Clasificador LGBM")
            proyecto = "ES_LGBM"
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_LGBM_ES(args.features)
            tracker.stop()
        elif args.clasificador == 'MLP':
            proyecto = "ES_MLP";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_MLP_ES(args.features)
            tracker.stop()
    if args.MW == "ANT":
        if args.clasificador == 'KNN':
            print("Clasificador KNN con N- " + str(args.vecinos) + " y NFeatures- " + str(args.features))
            proyecto = "ANT_KNN_K-" + str(args.vecinos) + "_NFeatures-" + str(args.features);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_HORMIGAS(args.vecinos,args.features, args.hormigas)
            tracker.stop()
        if args.clasificador == 'SVC':
            print("Clasificador svc con C- " + str(args.C))
            proyecto = "ANT_SVC_C-" + str(args.C);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_HORMIGAS(args.C)
            tracker.stop()
        if args.clasificador == 'XGBM':
            print("Clasificador XGBM")
            proyecto = "ANT_XBGM_NFEATURES-" + str(args.features) + "_ANTS" + str(args.hormigas)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_XGBM_HORMIGAS(args.features, args.hormigas)
            tracker.stop()
        if args.clasificador == 'LGBM':
            print("Clasificador LGBM")
            proyecto = "ANT_LBGM_NFEATURES-" + str(args.features) + "_ANTS" + str(args.hormigas)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_LGBM_HORMIGAS(args.features, args.hormigas)
            tracker.stop()
        if args.clasificador == 'MLP':
            print("Clasificador MLP")
            proyecto = "ANT_MLP_NFEATURES-" + str(args.features) + "_ANTS" + str(args.hormigas)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_MLP_HORMIGAS(args.features, args.hormigas)
            tracker.stop()
    if args.MW == "BEE":
        if args.clasificador == 'KNN':
            print("Clasificador KNN con K- " + str(args.vecinos) + " y NFeatures- " + str(
                args.features) + " ABEJAS:" + str(args.hormigas))
            proyecto = "BEE_KNN_K-" + str(args.vecinos) + "_NFeatures-" + str(args.features) + "_ANTS-" + str(
                args.hormigas)
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_KNN_BEE(args.vecinos,args.features)
            tracker.stop()
        if args.clasificador == 'SVC':
            print("Clasificador svc con C- " + str(args.C))
            proyecto = "BEE_SVC_C-" + str(args.C);
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_SVC_BEE(args.C, args.features)
            tracker.stop()
        if args.clasificador == 'XGBM':
            print("Clasificador XGBM")
            proyecto = "BEE_XBGM";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_XGBM_BEE(args.features)
            tracker.stop()
        if args.clasificador == 'LGBM':
            print("Clasificador LGBM")
            proyecto = "BEE_LBGM";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            clasificador_LGBM_BEE(args.features)
            tracker.stop()
        if args.clasificador == 'MLP':
            proyecto = "BEE_MLP";
            tracker = EmissionsTracker(project_name=proyecto)
            tracker.start()
            print("Clasificador MLP")
            clasificador_MLP_BEE(args.features)
            tracker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Selecciona un clasificador.')
    parser.add_argument('--clasificador', type=str, default='KNN',
                        choices=['KNN', 'SVC', 'MLP', 'XGBM', 'LGBM', 'ANT', 'BEE'],
                        help='Selecciona el clasificador (KNN o SVC).')
    parser.add_argument('--hormigas', type=int, default=12,
                        help='Selecciona el numero de hormigas.')
    parser.add_argument('--features', type=int, default=1500,
                        help='Selecciona el numero de features.')
    parser.add_argument('--vecinos', type=int, default=8,
                        help='Selecciona el numero de K-Vecinos.')
    parser.add_argument('--MW', type=str, default='GA', choices=['GA', 'PSO', 'ES','BEE','ANT'],
                        help='Selecciona el método wrapper')
    parser.add_argument('--C', type=float, default='1.0',
                        help='Selecciona el parametro C')
    parser.add_argument('--NGeneracion', type=int, default=300,
                        help='Selecciona el parametro Numero de Generaciones')
    parser.add_argument('--NGenes', type=int, default=1,
                        help='Selecciona el parametro Numero de genes')
    args = parser.parse_args()

    main(args)

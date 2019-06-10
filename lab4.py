import numpy as np
import itertools
import matplotlib.pyplot as plt 
import math

x = [[0, 0, 0, 0], #Вектор входных значений
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 1, 1],
     [0, 1, 0, 0],
     [0, 1, 0, 1],
     [0, 1, 1, 0],
     [0, 1, 1, 1],
     [1, 0, 0, 0],
     [1, 0, 0, 1],
     [1, 0, 1, 0],
     [1, 0, 1, 1],
     [1, 1, 0, 0],
     [1, 1, 0, 1],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

def RBF(): #Функция, определяющая РБФ нейроны
    J1 = [boolean_function(set) for set in x].count(1) #Находит количество наборов, на которых функция равна единице 
    if J1 <= 8:
        return [set for set in x if boolean_function(set)] #Если в векторе функции меньше "1", возвращаем наборы, на которых функция равна "1" 
    else:
        return [set for set in x if not boolean_function(set)] # -//- "0", -//- "0"

def learning(selection):
    nu = 0.3 #Норма обучения
    c = RBF() #Определяем центры
    u = np.zeros(len(c) + 1) #Вектор весов. Изначально заполняем нулями
    k = 0 #Эпоха обучения
    error_list = [] #Суммарные ошибки за все эпохи, по которым будет строиться график
    while not error_list or error_list[k - 1] > 0:
        print("Эпоха", k, ':')
        print("Веса:", np.around(u, 3))
        k += 1
        E = 0 #Суммарная ошибка в рамках одной эпохи
        y = [] #Выходной вектор
        for x in selection: 
            phi = get_phi(c, x) 
            net = np.dot(phi, u)
            out = heaviside_step_function(net)
            y.append(out)
            sigma = boolean_function(x) - out
            if sigma != 0:
                E += 1
                u += [nu * sigma * phi_i for phi_i in phi]
        print("Суммарная ошибка:", E)
        print("Выходной вектор:", y, "\n")
        error_list.append(E)

    if neuron_vector(c, u) == real_vector(): #Проверяем, можно ли на получившихся весах обучить НС
    #Это необходимо, когда нужно отыскать минимальный набор, на котором может обучиться НС
        print("НС может обучиться на выборке ", selection)
        plot(k, error_list)
        return k
    else:
        print("НС не может обучиться на выборке ", selection, '\n')
        return 0


def heaviside_step_function(net): #Пороговая функция активации
    if net >= 0:
        return 1
    else:
        return 0

def plot(k, list_errors): #Функция, строящая график зависимости ошибки от эпохи
    X = [i for i in range(k)]
    Y = [list_errors[i] for i in X]
    plt.plot(X, Y)
    plt.xlabel('Eras')
    plt.ylabel('Errors')
    plt.show()

def neuron_vector(c, u): #Булевый вектор, который высчитал нейрон, используя веса, которые он получил, обучаясь на какой-то выборке
    out = ''
    for set in x:
        phi = get_phi(c, set)
        net = np.dot(phi, u)
        y = heaviside_step_function(net)
        out += str(y)
    return out

def real_vector(): #Вектор булевой функции
    real_vector = ''
    for set in x:
        real_vector += str(boolean_function(set))
    return real_vector

def get_phi(c, x): #Значения в RBF-нейронах
    difference = [[np.array(x) - np.array(index)] for index in c]
    return [1] + [math.exp(-np.sum(np.power(index, 2))) for index in difference]


def boolean_function(x): #Функция, высчитывающая значения булевой функции
    if ((x[0]+x[1]+x[2])*(x[1]+x[2]+x[3])) > 0:
        return 1
    else:
        return 0

def selection(): #Поиск обучающей выборки
    for i in range(1, 16):
            for sets in itertools.permutations(x, i):
                print("Выборка:_______________", sets,"_______________")
                if learning(list(sets)):
                    return None

learning(x)
selection()

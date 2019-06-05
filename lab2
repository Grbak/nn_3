import numpy as np
import matplotlib.pyplot as plt
import math


def Train(p, eras, nu, isplot = True): #Функция, обучающая нейронную сеть. p - длина окна, ages - количество эпох, nu - норма обучения
    array = np.linspace(-1, 2, 20)
    matrix = [[Function(array[i + j]) for j in range(p)] for i in range(20 - p)]
    w = np.zeros(p)
    era = 0
    for _ in range(eras):
        error = 0
        era+=1
        for i in range(len(matrix)):
            sigma = Function(array[p + i]) - np.dot(w,matrix[i])
            error += sigma ** 2
            w += [nu * sigma * k for k in matrix[i]]
        error = error ** 0.5
        print('Эпоха:', era)
        print(np.around(w, 3))
    #print(f'Era:{eras} nu = {np.around(nu, 1)} p = {p}\nVector w:{np.around(w, 3)}')
    return Plot(w, p) if isplot else w

def Real_Plot():
    x = np.linspace(-1,2, 20)
    plt.plot(x, [Function(t) for t in x], 'c-')
    plt.grid()
    plt.show()


def Plot(w, p):
    x = np.linspace(-1, 5, 40)
    matrix = np.array([[Function(x[i + j]) for j in range(p)] for i in range(20-p, 40 - p)])
    plt.plot(x, [Function(t) for t in x], 'c-')
    plt.plot(x[20:], [np.dot(w, arr) for arr in matrix], 'r-')
    print(f'Error: {np.around(Get_error(w, p), 4)}')
    plt.grid()
    plt.show()

def Get_error(w, p): #Функция, вычисляющая ошибки для построение зависимостей
    x = np.linspace(-1, 5, 40)
    matrix = np.array([[Function(x[i + j]) for j in range(p)] for i in range(20-p, 40 - p)])
    Real_line = np.array([Function(t) for t in x])
    Neuron_line = [np.dot(w, arr) for arr in matrix]
    error = Real_line[20:]-Neuron_line
    return sum([a**2 for a in error])

def Function(t): #Заданная функция
    return np.e**(t-2) - math.sin(t)

def Test_nu(): #Функция, строящая график зависимости ошибки от нормы обучения
    error_list = []
    for i in np.arange(0.1, 1.1, 0.1):
        w = Train(6, 2000, i, False)
        error_list.append(Get_error(w, 6))
    plt.plot(np.arange(0.1, 1.1, 0.1), error_list,'c-')
    plt.title("Зависимость ошибки от нормы обучения")
    plt.xlabel("Норма обучения")
    plt.ylabel("Ошибка")
    plt.grid()
    plt.show()

def Test_p(): #Функция, строящая график зависимости ошибки от длины окна
    error_list = []
    for i in np.arange(1, 17):
        w = Train(i, 500, 0.4, False)
        error_list.append(Get_error(w, i))
    plt.plot(np.arange(1, 17), error_list,'c-')
    plt.title("Зависимость ошибки от длины окна")
    plt.xlabel("Длина окна")
    plt.ylabel("Ошибка")
    plt.grid()
    plt.show()

def Test_era(): #Функция, строящая график зависимости ошибки от количества эпох
    error_list = []
    for i in np.arange(500, 2500, 500):
        w = Train(6, i, 0.4, False)
        error_list.append(Get_error(w, 6))
    plt.plot(np.arange(500, 2500, 500), error_list,'c-')
    plt.title("Зависимость ошибки от количества эпох")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.grid()
    plt.show()

        

Real_Plot()
Train(4, 500, 1)
Train(4, 1500, 1)
Test_nu()
Test_p()
Test_era()

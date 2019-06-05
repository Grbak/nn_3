import numpy as np
import math
import itertools

nu = 1 #Норма обучения
eps = 0.001 #Порог, который суммарная ошибка не должна превышать

def sigmoid_function(net): #Функция активации
	return (1 - np.e**(-net))/(1 + np.e**(-net))

def derivative(net): #Производная функции активации
	return 0.5*(1 - sigmoid_function(net)**2)

class Neuron: #Класс нейрона
	def __init__(self, numb_w):
		self.w = np.array([0.5]*numb_w) #Задаем вектор весов длиной numb_w, начальные веса возьмем равными 0.5
		self.net = 0
		self.sigma = 0

	def get_out(self, x): #Функция, высчитывающая выход нейрона
		self.net = np.dot(self.w, x)
		return sigmoid_function(self.net)

	def correct_weight(self, x): #Функция, корректирующая веса
		self.w +=[ nu*derivative(self.net)*self.sigma*xi for xi in x]
		self.net = 0
		self.sigma = 0

	def get_mistake(self, sigma): #Функция, высчитывающая ошибку
		self.sigma = derivative(self.net)*sigma
		return [self.sigma*wi for wi in self.w[1:]] #Функция возвращает такую величину для удобства: впоследствии мы используем это, когда будем высчитывать ошибку для нейронов скрытого слоя


class Neuronal_network: #Класс нейронной сети
	def __init__(self, x, J, t):
		self.x = x #Вектор входных данных нашей нейронной сети
		self.J = J #Количество нейронов скрытого слоя
		self.t = t #Вектор целевого выхода
		self.age = 0 #Эпоха обучения

		self.hide_level = [Neuron(len(x)) for index in range(J)] #Создаем вектор нейронов скрытого слоя
		self.output_level = [Neuron(J+1) for index in range(len(t))] # -//- выходного слоя

	def Age(self): #Одна эпоха обучения
		print('_________________Эпоха', self.age, '__________________')

		hide_out = [1] + [hide_neuron.get_out(self.x) for hide_neuron in self.hide_level] #Cоздаем вектор выходов нейронов скрытого слоя
		#"[1]" - нейрон смещения, который потребуется, когда мы будем подавать этот вектор на выходной слой
		output_out = [output_neuron.get_out(hide_out) for output_neuron  in self.output_level] # -//- выходного слоя


		sigma = [self.t[index] - output_out[index] for index in range(len(self.t))] #Для каждого нейрона выходного слоя высчитываем ошибку

		squared_error =  sum(index**2 for index in sigma)**0.5 #Суммарная среднеквадратичная ошибка

		print("Веса нейронов скрытого слоя:")
		for neuron in self.hide_level:
			print(np.around(neuron.w, 3))

		print("Веса нейронов выходного слоя:")
		for neuron in self.output_level:
			print(np.around(neuron.w, 3))

		print("Выходы нейронов выходного слоя:")
		print(np.around(output_out, 3))

		print('Суммарная среднеквадратичная ошибка:', round(squared_error, 4))


		output_mistake = [output_neuron.get_mistake(sigma[i]) for i, output_neuron in enumerate(self.output_level)] #Для каждого нейрона выходного слоя вызываем метод get_mistake, используя соответствующий элемент вектора sigma
		output_mistake = [sum(composition[index] for composition in output_mistake) for index in range(self.J)] #Складываем все элементы вектора output_mistake. Данная сумма необходима для вычисления ошибки на нейронах скрытого слоя

		for hide_neuron, j in itertools.product(self.hide_level, range(len(output_mistake))): #Для каждого нейрона скрытого слоя вычисляем ошибку
			hide_neuron.get_mistake(output_mistake[j])

		for hide_neuron in self.hide_level:
			hide_neuron.correct_weight(self.x)

		for output_neuron in self.output_level:
			output_neuron.correct_weight(hide_out)

		self.age += 1

		return squared_error

	def Learning(self):
 		while self.Age() > eps:
 			pass

n1 = Neuronal_network([1, 2, 2], 1 , [0.3, 0.1])
n1.Learning()

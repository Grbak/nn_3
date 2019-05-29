import numpy as np
import math
import itertools

nu = 0.5 #Норму обучения возьмем равной 0.5
eps = 0.0001 #Порог, который суммарная ошибка не должна превышать

def Function(net): #Функция активации
	return (1 - np.e**(-net))/(1 + np.e**(-net))

def Derivative(net): #Производная функции активации
	return 0.5*(1 - Function(net)**2)


class N: #Класс нейрона

	def __init__(self, numb_w):
		self.w = np.array([0.5]*numb_w) #Задаем вектор весов длиной numb_w, начальные веса возьмем равными 0.5
		self.net = 0 
		self.sygma = 0

	def Get_out(self, x): #Функция, высчитывающая выход нейрона
		self.net = np.dot(self.w, x) 
		return Function(self.net)

	def Correct_weight(self, x): #Функция, корректирующая веса
		self.w +=[ nu*self.sygma*xi for xi in x] 
		self.net = 0 
		self.sygma = 0

	def Get_mistake(self, sygma): #Функция, высчитывающая ошибку
		self.sygma = Derivative(self.net)*sygma
		return [self.sygma*wi for wi in self.w[1:]] #Функция возвращает такую величину для удобства: впоследствии мы используем это, когда будем высчитывать ошибку для нейронов скрытого слоя




class NN: #Класс нейронной сети
	def __init__(self, x, J, t):
		self.x = x #Вектор входных данных нашей нейронной сети
		self.J = J #Количество нейронов скрытого слоя
		self.t = t #Вектор целевого выхода

		self.age = 0 #Эпоха обучения

		self.hide_level = [N(len(x)) for index in range(J)] #Создаем вектор нейронов скрытого слоя
		self.output_level = [N(J+1) for index in range(len(t))] # -//- выходного слоя

	def Age(self): #Одна эпоха обучения
		print('_________________Эпоха', self.age, '__________________')

		hide_out = [1] + [hide_neuron.Get_out(self.x) for hide_neuron in self.hide_level] #Cоздаем вектор выходов нейронов скрытого слоя
		#"[1]" - нейрон смещения, который потребуется, когда мы будем подавать этот вектор на выходной слой
		output_out = [output_neuron.Get_out(hide_out) for output_neuron  in self.output_level] # -//- выходного слоя

		sygma = [self.t[index] - output_out[index] for index in range(len(self.t))] #Для каждого нейрона выходного слоя высчитываем ошибку


		squared_mistake =  sum(index**2 for index in sygma)**0.5 #Суммарная среднеквадратичная ошибка

		print('Суммарная среднеквадратичная ошибка:', round(squared_mistake, 5))

		print("Веса нейронов скрытого слоя:")
		for hide_neuron in self.hide_level:
			print(np.around(hide_neuron.w, 3))

		print("Веса нейронов выходного слоя:")
		for output_neuron in self.output_level:
			print(np.around(output_neuron.w, 3))

		output_mistake = [output_neuron.Get_mistake(sygma[index]) for index, output_neuron in enumerate(self.output_level)] #Для каждого нейрона выходного слоя вызываем метод Get_mistake, используя соответствующий элемент вектора sygma
		output_mistake = [sum(composition[index] for composition in output_mistake) for index in range(self.J)]	#Складываем все элементы вектора output_mistake. Данная сумма необходима для вычисления ошибки на нейронах скрытого слоя

		for hide_neuron, index in itertools.product(self.hide_level, range(len(output_mistake))): #Для каждого нейрона скрытого слоя вычисляем ошибку
			hide_neuron.Get_mistake(output_mistake[index]) 


		for hide_neuron in self.hide_level:
			hide_neuron.Correct_weight(self.x)


		for output_neuron in self.output_level:
			output_neuron.Correct_weight(hide_out)


		self.age += 1

		return squared_mistake

	def Learning(self):
 		while self.Age() > eps:
 			pass


n1 = NN([1, 2, 2], 1 , [0.3, 0.1])
n1.Learning()

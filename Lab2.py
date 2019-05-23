import numpy as np
import math
import itertools

nu = 0.5
eps = 0.0001

def Function(net):
	return (1 - np.e**(-net))/(1 + np.e**(-net))

def Derivative(net):
	return 0.5*(1 - Function(net)**2)

class N:
	def __init__(self, numb_w):
		self.w = np.array([0.5]*numb_w)
		self.net = 0
		self.sygma = 0

	def Get_out(self, x):
		self.net = np.dot(self.w, x)

		return Function(self.net)

	def Correct_weight(self, x):
		self.w +=[ nu*Derivative(self.net)*self.sygma*xi for xi in x]
		self.net = 0
		self.sygma = 0

	def Get_mistake(self, sygma):
		self.sygma = Derivative(self.net)*sygma
		return [self.sygma*wi for wi in self.w[1:]]

class NN:
	def __init__(self, x, J, t):
		self.x = x
		self.J = J
		self.t = t
		self.age = 0

		self.hide_level = [N(len(x)) for index in range(J)]
		self.output_level = [N(J+1) for index in range(len(t))]

	def Age(self):
		print('_________________Эпоха', self.age, '__________________')

		hide_out = [1] + [hide_neuron.Get_out(self.x) for hide_neuron in self.hide_level]

		output_out = [output_neuron.Get_out(hide_out) for output_neuron  in self.output_level]

		sygma = [self.t[index] - output_out[index] for index in range(len(self.t))]


		squared_error =  sum(index**2 for index in sygma)**0.5

		print('Суммарная среднеквадратичная ошибка:', round(squared_error, 5))

		print("Веса нейронов скрытого слоя:")
		for neuron in self.hide_level:
			print(np.around(neuron.w, 3))

		print("Веса нейронов выходного слоя:")
		for neuron in self.output_level:
			print(np.around(neuron.w, 3))

		output_mistake = [output_neuron.Get_mistake(sygma[i]) for i, output_neuron in enumerate(self.output_level)]
		output_mistake = [sum(composition[index] for composition in output_mistake) for index in range(self.J)]

		for hide_neuron, j in itertools.product(self.hide_level, range(len(output_mistake))):
			hide_neuron.Get_mistake(output_mistake[j])

		for hide_neuron in self.hide_level:
			hide_neuron.Correct_weight(self.x)

		for output_neuron in self.output_level:
			output_neuron.Correct_weight(hide_out)

		self.age += 1

		return squared_error

	def Learning(self):
 		while self.Age() > eps:
 			pass

n1 = NN([1, 2, 2], 1 , [0.3, 0.1])
n1.Learning()

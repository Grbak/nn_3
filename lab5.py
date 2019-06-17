import numpy as np

values = {1: '1', -1: '0'}

x0 = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
x1 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1] #2
x2 = [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1] #4
x3 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1] #6
x = [x1, x2, x3]

class Hopfield_network:
    def __init__(self, I, J):
        self.models = []
        self.J = J
        self.I = I
        self.n = self.J*self.I
        self.w = np.array([np.zeros(self.n) for i in range(self.n)])

    def training_mode(self, x):
        self.models.append(x)

        for i in range(self.n):
            for j in range(self.n):
                if(i == j):
                    self.w[i][j] = 0
                else:
                    self.w[i][j] += x[i] * x[j]

    def FuncActiv(self, net, y):
        if net > 0:
            return 1
        elif net == 0:
            return 0
        else:
            return -1
    
    def net(self, x):
        for i in range(self.n):
            net_y = sum([self.w[j][i] * x[j] for j in range(self.n-1)])
            y = self.FuncActiv(net_y, x[i])
            if y != x[i] and y != 0:
                print("Изменение", i,'-го бита')
                x[i] = y

        if x not in self.models:
            print("Распознования не произошло!")
            return 0
        return x
        
    def parse(self, x):
        models = []
        for a in x:
            models.append(self.parseModel(a))
        return models

    def parseModel(self, x):       
        models = []
        for c in x:
            if c == 1:
                models.append(1)
            if c == 0:
                models.append(-1)
        return models

    def get_Etalon(self, obraz):
        for_print = "".join([values[a] for a in obraz])
        for i in range(self.J):
            print(for_print[i * self.I: i * self.I + self.I])
        print('')


if __name__ == '__main__':

    N = Hopfield_network(3, 5)

    etalons = N.parse(x)
    test_etalon = N.parseModel(x0)

    print("Образы, которые нейронная сеть может распознать:")

    for e in etalons:
        N.get_Etalon(e)

    for e in etalons:
        N.training_mode(e)

    print("Веса РНС Хопфилда в векторно-матричном виде:\n")
    print(N.w, "\n")
    print("Образ, поданый нейронной сети:\n")
    N.get_Etalon(test_etalon)
    new_model = N.net(test_etalon)
    if new_model == 0:
        exit(0)
    print("\n")
    print("Распознование прошло успешно!")
    N.get_Etalon(new_model)

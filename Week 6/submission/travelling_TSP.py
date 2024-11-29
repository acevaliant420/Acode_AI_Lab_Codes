import numpy as np
import matplotlib.pyplot as plt

class ModifiedHopfield:
    def __init__(self, *, number_of_objects, alpha, sigma, **kwargs):
        assert number_of_objects > 0
        self.number_of_objects = number_of_objects
        self.alpha = alpha
        self.sigma = sigma
        self.constants = kwargs
        self.input_potential_shape = (self.number_of_objects,) * 2

    def solution(self):
        return self.output_potential > 0

    def output_potential_function(self, input_potential):
        return (1. + np.tanh(self.alpha * input_potential)) / 2

    def energy_function(self, *args):
        E = 0
        E_1 = 0
        for x in range(self.number_of_objects):
            for i in range(self.number_of_objects):
                for j in range(self.number_of_objects):
                    if i != j:
                        E_1 += self.output_potential[x, i] * self.output_potential[x, j]
        E += self.constants["A"] / 2 * E_1

        E_2 = 0
        for i in range(self.number_of_objects):
            for x in range(self.number_of_objects):
                for y in range(self.number_of_objects):
                    if x != y:
                        E_2 += self.output_potential[x, i] * self.output_potential[y, i]
        E += self.constants["B"] / 2 * E_2

        E_3 = (self.output_potential.sum() - (self.number_of_objects + self.sigma)) ** 2
        E += self.constants["C"] / 2 * E_3

        E_4 = 0
        for x in range(self.number_of_objects):
            for y in range(self.number_of_objects):
                if x != y:
                    E_4 += args[0][x][y] * self.output_potential[x, 0] * self.output_potential[y, 1]
                    for i in range(1, self.number_of_objects - 1):
                        E_4 += args[0][x][y] * self.output_potential[x, i] * (
                                self.output_potential[y, i + 1] + self.output_potential[y, i - 1])
                    E_4 += args[0][x][y] * self.output_potential[x, self.number_of_objects - 1] * \
                           self.output_potential[y, self.number_of_objects - 2]
        E += self.constants["D"] / 2 * E_4

        return E
    def count_weights(self):
 
        weights = {
            'inhibition_same_row': 0,    
            'inhibition_same_column': 0,  
            'path_constraints': 0,       
            'total': 0
        }
        
        for x in range(self.number_of_objects):
            for i in range(self.number_of_objects):
                for j in range(self.number_of_objects):
                    if i != j:
                        weights['inhibition_same_row'] += 1
                        
        for i in range(self.number_of_objects):
            for x in range(self.number_of_objects):
                for y in range(self.number_of_objects):
                    if x != y:
                        weights['inhibition_same_column'] += 1
                        
        for x in range(self.number_of_objects):
            for y in range(self.number_of_objects):
                if x != y:
                    weights['path_constraints'] += 1
                    weights['path_constraints'] += 2 * (self.number_of_objects - 2)
                    weights['path_constraints'] += 1
        
        weights['total'] = (weights['inhibition_same_row'] + 
                        weights['inhibition_same_column'] + 
                        weights['path_constraints'])
        
        return weights
    def train(self, *args, max_external_iterations=25, max_external_iterations_with_constant_E=5, strategy="A",
              plot_energy_graph=False):
        assert len(args) == 0 or (
                    len(args[0]) == self.number_of_objects and all(
                [len(args[0][i]) == self.number_of_objects for i in range(self.number_of_objects)]))
        internal_iterations, external_iterations, external_iterations_with_constant_E = 0, 0, 0
        self.input_potential = np.zeros(self.input_potential_shape)
        self.output_potential = self.output_potential_function(self.input_potential)
        if strategy == "A":
            self.output_potential = np.random.uniform(0, 0.03, self.input_potential_shape)
        elif strategy == "B":
            self.output_potential = np.random.uniform(0, 1, self.input_potential_shape)
        elif strategy == "C":
            self.output_potential = np.random.uniform(1 - 0.03, 1, self.input_potential_shape)
        elif strategy == "D":
            self.output_potential = np.random.uniform(0, 0.03, self.input_potential_shape) + 1 / self.number_of_objects
        energy_list = [self.energy_function(args[0])]
        while external_iterations_with_constant_E < max_external_iterations_with_constant_E and external_iterations < max_external_iterations:
            for _ in range(self.number_of_objects ** 2):
                x, i = np.random.randint(self.number_of_objects), np.random.randint(self.number_of_objects)
                self.input_potential[x, i] = - (self.constants["A"] * (self.output_potential[x].sum() - self.output_potential[x, i]) +
                                                self.constants["B"] * (self.output_potential[:, i].sum() - self.output_potential[x, i]) +
                                                self.constants["C"] * (self.output_potential.sum() - (self.number_of_objects + self.sigma)))
                if i == 0:
                    self.input_potential[x, i] -= self.constants["D"] * np.matmul(args[0][x], self.output_potential[:, 1])
                elif i == self.number_of_objects - 1:
                    self.input_potential[x, i] -= self.constants["D"] * np.matmul(args[0][x],
                                                                                  self.output_potential[:, self.number_of_objects - 2])
                else:
                    self.input_potential[x, i] -= self.constants["D"] * np.matmul(args[0][x],
                                                                                  self.output_potential[:, i + 1] + self.output_potential[:, i - 1])
                self.output_potential[x, i] = self.output_potential_function(self.input_potential[x, i])
            internal_iterations += 1
            if internal_iterations % 5 == 0:
                external_iterations += 1
                energy = self.energy_function(args[0])
                if energy == energy_list[-1]:
                    external_iterations_with_constant_E += 1
                else:
                    external_iterations_with_constant_E = 0
                energy_list.append(energy)
        if plot_energy_graph:
            plt.plot([i for i in range(external_iterations + 1)], energy_list)
            plt.xlabel("Number of External Iterations")
            plt.ylabel("Energy")
            plt.annotate(f"min(E) = {int(energy_list[-1])}", (len(energy_list) - 1 + 0.01, energy_list[-1] + 0.01))
            plt.show()
        return energy_list[-1], internal_iterations, external_iterations


cities = [(0.25, 0.16), (0.85, 0.35), (0.65, 0.24), (0.70, 0.50), (0.15, 0.22), (0.25, 0.78), (0.40, 0.45), (0.90, 0.65),
          (0.55, 0.90), (0.60, 0.28)]
number_of_cities = len(cities)
distances = [[((city_2[0] - city_1[0]) ** 2 + (city_2[1] - city_1[1]) ** 2) ** 0.5 for city_2 in cities] for city_1 in
             cities]
labels = [chr(65 + i) for i in range(number_of_cities)]

plt.figure(figsize=(8, 8))
plt.scatter(*zip(*cities))
for i in range(number_of_cities):
    plt.annotate(labels[i], (cities[i][0] + 0.01, cities[i][1] + 0.01))
plt.show()

A, B, C, D, sigma = 100, 100, 90, 100, 1
alpha = 50
np.random.seed(123)

hopfield_network = ModifiedHopfield(number_of_objects=number_of_cities,
                                    alpha=alpha,
                                    sigma=sigma,
                                    A=A,
                                    B=B,
                                    C=C,
                                    D=D)

hopfield_network.train(distances,
                       max_external_iterations=25,
                       max_external_iterations_with_constant_E=5,
                       plot_energy_graph=True)


weights = hopfield_network.count_weights()
print("Weight counts:")
for component, count in weights.items():
    print(f"{component}: {count}")

    
solution = list(np.take(np.asarray(cities), np.where(hopfield_network.solution().T)[1], axis=0))
solution.append(solution[0])
plt.figure(figsize=(8, 8))
plt.plot(*zip(*solution), "-o")
for i in range(number_of_cities):
    plt.annotate(labels[i], (cities[i][0] + 0.01, cities[i][1] + 0.01))
if len(solution) > number_of_cities:
    total_distance = sum(
        [((solution[i][0] - solution[i + 1][0]) ** 2 + (solution[i][1] - solution[i + 1][1]) ** 2) ** 0.5 for i in
         range(number_of_cities)])
else:
    total_distance = None
plt.title(f"Distance travelled: {total_distance}")
plt.show()

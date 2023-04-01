import numpy as np
import matplotlib.pyplot as plt
from numpy import sum as numpysum
from numpy import asarray


def dejong2(x):
    assert len(x) >= 2
    #x = np.asarray(x)
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
#    return np.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2 - 0.3 * np.cos(3 * np.pi * x[:-1]) - 0.4 * np.cos(4 * np.pi * x[1:]))


# Nastavení rozměru prostoru a počtu iterací
dim = 2
num_iterations = 10000

# Nastavení hranic prostoru
bounds = np.array([[-5, 5]] * dim)

# Inicializace nejlepšího řešení a jeho fitness
best_solution = None
best_fitness = np.inf

# Hlavní cyklus algoritmu
for i in range(num_iterations):
    # Vygenerování náhodného řešení v daných hranicích
    candidate_solution = np.random.uniform(bounds[:, 0], bounds[:, 1])
    candidate_fitness = dejong2(candidate_solution)

    # Porovnání fitness s nejlepším řešením
    if candidate_fitness < best_fitness:
        best_solution = candidate_solution
        best_fitness = candidate_fitness

    # Výpis nejlepšího řešení v každé desáté iteraci
    if i % 100 == 0:
    # if i > 1:
        print(f"Iteration {i}: Best solution found: {best_solution}, fitness: {best_fitness}")

# Výpis výsledku
print(f"Best solution found: {best_solution}, fitness: {best_fitness}")

# Vykreslení funkce
resolution = 100
x = np.linspace(bounds[0][0], bounds[0][1], resolution)
y = np.linspace(bounds[1][0], bounds[1][1], resolution)
X, Y = np.meshgrid(x, y)
Z = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        Z[i][j] = dejong2([X[i][j], Y[i][j]])
plt.contour(X, Y, Z, levels=np.logspace(-0.5, 5, 50))
plt.plot(best_solution[0], best_solution[1], 'r*', markersize=10)
plt.savefig(f"C:/Users/vnovotny/Desktop/Garbage/dejong2_random_search.png")
plt.show()
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# definice funkce DeJong1 pro pět dimenzí
def dejong1(x):
    return sum([xi**2 for xi in x])

# define 2nd Dejong function
def dejong2(x):
    assert len(x) >= 2
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# define Schwefel function
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

# nastavení počátečních hodnot
x0 = [random.uniform(-5.12, 5.12) for i in range(5)]
T0 = 100
Tf = 0.01
alfa = 0.9
num_runs = 30

# funkce pro generování náhodného souseda
def generate_neighbor(x):
    i = random.randint(0, len(x)-1)
    direction = 1 if random.random() < 0.5 else -1
    delta = random.uniform(0, 1)
    new_x = x.copy()
    new_x[i] += direction * delta
    return new_x

# inicializace proměnných
x_current = x0
f_current = dejong1(x_current)
T = T0
best_solution = x_current.copy()

# simulované žíhání
for run in range(num_runs):
    while T > Tf:
        for i in range(len(x0)):
            x_new = generate_neighbor(x_current)
            f_new = dejong1(x_new)
            delta_f = f_new - f_current
            if delta_f < 0:
                x_current = x_new
                f_current = f_new
                if f_current < dejong1(best_solution):
                    best_solution = x_current.copy()
            else:
                p_accept = math.exp(-delta_f / T)
                if random.random() < p_accept:
                    x_current = x_new
                    f_current = f_new
        T *= alfa

    # resetování proměnných pro nový běh
    x_current = x0
    f_current = dejong1(x_current)
    T = T0

print("Best solution:", best_solution)
print("Objective function value:", dejong1(best_solution))
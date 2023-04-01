import random
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import sum as numpysum


def dejong1(x):
    return sum([xi ** 2 for xi in x])


def dejong2(x):
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def schwefel(x):
    n = len (x)
    return 418.9829 * n - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


# Define the bounds of the search space
bounds_dj1_D5 = np.array([[-5, 5]] * 5)
bounds_dj2_D5 = np.array([[-5, 5]] * 5)
bounds_swf_D5 = np.array([[-500, 500]] * 5)

bounds_dj1_D10 = np.array([[-5, 5]] * 10)
bounds_dj2_D10 = np.array([[-5, 5]] * 10)
bounds_swf_D10 = np.array([[-500, 500]] * 10)

global_dj1_D5 = np.full(5, 0)
global_dj2_D5 = np.full(5, 1)
global_swf_D5 = np.full(5, 420.9687)
global_dj1_D10 = np.full(10, 0)
global_dj2_D10 = np.full(10, 1)
global_swf_D10 = np.full(10, 420.9687)


def random_search(f, bounds, max_iter):
    global best_fitness
    global best_result
    for i in range(max_iter):
        # Generate a random candidate solution within the search space
        candidate = [np.random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]

        # Evaluate the candidate solution
        fitness = f(candidate)

        # Update the best solution if necessary
        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness
            best_result = candidate

    return best_result, best_fitness


    print(f"Best solution after 30 runs: {best_result}")


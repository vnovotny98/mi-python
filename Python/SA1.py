import math
import random
import numpy as np
import matplotlib.pyplot as plt

# define 1st Dejong function
def dejong1(x):
    return sum(xi ** 2 for xi in x)


# define 2nd Dejong function
def dejong2(x):
    assert len(x) >= 2
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# define Schwefel function
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def simulated_annealing(f, bounds, initial_temp, final_temp, max_iter):
    current_solution = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
    current_fitness = f(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    for i in range(max_iter):
        # Calculate the temperature for this iteration
        temperature = initial_temp * ((final_temp / initial_temp) ** (i / (max_iter - 1)))

        # Generate a candidate solution by randomly perturbing the current solution
        candidate = [current_solution[j] + random.uniform(-1, 1) * temperature for j in range(len(bounds))]

        # Clip the candidate solution to the search space bounds
        candidate = [min(max(candidate[j], bounds[j][0]), bounds[j][1]) for j in range(len(bounds))]

        # Evaluate the candidate solution
        candidate_fitness = f(candidate)

        # Accept the candidate solution if it is better than the current solution, or with a certain probability if it is worse
        if candidate_fitness < current_fitness or math.exp(
                (current_fitness - candidate_fitness) / temperature) > random.uniform(0, 1):
            current_solution = candidate
            current_fitness = candidate_fitness

            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

    return best_solution, best_fitness


# Define the bounds of the search space
bounds = [(-5.12, 5.12)] * 2

# Perform the simulated annealing
best_solution, best_fitness = simulated_annealing(dejong2, bounds, initial_temp=100, final_temp=0.1, max_iter=1000)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
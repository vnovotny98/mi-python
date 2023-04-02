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


def simulated_annealing(f, bounds, max_iter, initial_temp, cooling_rate):
    # Generate a random initial solution within the search space
    current_solution = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
    current_fitness = f(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    # Loop through the specified number of iterations
    for i in range(max_iter):
        # Calculate the current temperature
        temperature = initial_temp * math.exp(-cooling_rate * i)

        # Generate a new candidate solution within the search space
        candidate_solution = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
        candidate_fitness = f(candidate_solution)

        # Determine if the new solution should be accepted
        delta_fitness = candidate_fitness - current_fitness

        if delta_fitness < 0:
            current_solution = candidate_solution
            current_fitness = candidate_fitness
        elif temperature == 0:
            continue
        else:
            try:
                if math.exp(-delta_fitness / temperature) > random.uniform(0, 1):
                    current_solution = candidate_solution
                    current_fitness = candidate_fitness
            except OverflowError:
                current_solution = candidate_solution
                current_fitness = candidate_fitness

    return best_solution, best_fitness


# Define the bounds of the search space
bounds = [(-5, 5)] * 2

# Perform the simulated annealing search
best_solution, best_fitness = simulated_annealing(dejong2, bounds, max_iter=1000, initial_temp=10, cooling_rate=0.83)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
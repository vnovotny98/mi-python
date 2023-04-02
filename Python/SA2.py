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


def simulated_annealing(f, bounds, max_iter, initial_temperature, cooling_rate):
    # Initialize the current solution
    current_solution = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
    current_fitness = f(current_solution)

    # Initialize the best solution
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    # Initialize the temperature
    temperature = initial_temperature

    # Perform the simulated annealing
    for i in range(max_iter):
        # Generate a random candidate solution
        candidate_solution = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
        candidate_fitness = f(candidate_solution)

        # Calculate the acceptance probability
        delta_fitness = candidate_fitness - current_fitness
        acceptance_probability = math.exp(-delta_fitness / temperature)

        # Determine whether to accept the candidate solution
        if delta_fitness < 0 or random.random() < acceptance_probability:
            current_solution = candidate_solution
            current_fitness = candidate_fitness

            # Update the best solution if necessary
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

        # Lower the temperature
        temperature *= cooling_rate

    return best_solution, best_fitness


# Define the bounds of the search space
bounds = [(-5, 5)] * 5

# Perform the simulated annealing
best_solution, best_fitness = simulated_annealing(dejong2, bounds, 1000, 100, 0.95)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
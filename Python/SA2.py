import random
import math
from numpy import asarray
from numpy import sum as numpysum


#def dejong2(x):
#    return sum([(i + 1) * xi ** 2 for i, xi in enumerate(x)])

def dejong2(x):
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


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
bounds = [(-5, 5)] * 2

# Perform the simulated annealing
best_solution, best_fitness = simulated_annealing(dejong2, bounds, 1000, 100, 0.95)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
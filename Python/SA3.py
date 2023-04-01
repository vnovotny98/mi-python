import math
import random
from numpy import asarray
from numpy import sum as numpysum

def dejong2(x):
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


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
        # if delta_fitness < 0 or math.exp(-delta_fitness / temperature) > random.uniform(0, 1):
        #    current_solution = candidate_solution
        #    current_fitness = candidate_fitness
        if delta_fitness < 0 or (temperature > 0 and math.exp(-delta_fitness / temperature) > random.uniform(0, 1)):
            current_solution = candidate_solution
            current_fitness = candidate_fitness

        # Update the best solution if necessary
        if current_fitness < best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

    return best_solution, best_fitness


# Define the bounds of the search space
bounds = [(-5.12, 5.12)] * 2

# Perform the simulated annealing search
best_solution, best_fitness = simulated_annealing(dejong2, bounds, max_iter=1000, initial_temp=100, cooling_rate=0.95)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
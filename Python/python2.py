import random
import math


def dejong1(x):
    return sum([xi ** 2 for xi in x])

print("Best solution")

def random_search(f, bounds, max_iter):
    best_result = None
    best_fitness = None
    for i in range(max_iter):
        # Generate a random candidate solution within the search space
        candidate = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]

        # Evaluate the candidate solution
        fitness = f(candidate)

        # Update the best solution if necessary
        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness
            best_result = candidate

    return best_result, best_fitness


# Define the bounds of the search space
bounds = [(-5.12, 5.12)] * 2
max_iter = 1000

# Perform the random search
best_solution, best_fitness = random_search(dejong1, bounds, max_iter)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
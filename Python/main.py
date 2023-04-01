import numpy as np
import funcs as algo_fun
import matplotlib.pyplot as plt

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

max_iter = 10000

# Perform the random search
best_solution, best_fitness = algo_fun.random_search(algo_fun.dejong1, bounds_dj1_D5, max_iter)

# Print the results
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

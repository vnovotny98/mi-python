import random
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import sum as numpysum

_pop_size = 10

def dejong1(x):
    return sum([xi ** 2 for xi in x])

def dejong2(x):
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def schwefel(x):
    n = len (x)
    return 418.9829 * n - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])

def random_search(obj_func, bounds, max_iter, pop_size=_pop_size):
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

    fitness = [obj_func(ind) for ind in pop]

    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]

    for _ in range(max_iter):
        new_pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, bounds.shape[0]))

        fitness = [obj_func(ind) for ind in new_pop]

        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness
        best_solution = pop

        return best_solution, best_fitness



plt.xlabel("Number of iterations", fontsize=20)
plt.ylabel("Fitness", fontsize=20)
plt.text(0.05, 0.95, f"Best solution: {best_solution}", transform.plt.gca().transAxes, va='top', fontsize=18)

def simulated_annealing(obj_func, bounds, max_iter, pop_size=_pop_size, temperature=100, cooling_rate=0.95):
    # Initialize the population and fitness arrays
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size= (pop_size, bounds.shape[0]))
    fitness = np.array([obj_func(ind) for ind in pop])
    # Set the initial best fitness and solution
    best_fitness = np.min(fitness)
    best_solution = pop[np.argmin(fitness)]
    #Initialize the acceptance probabilities array
    acceptace_probs = np.zeros(pop_size)



plt.figure(figsize=(24, 12))

#algo_fun.plot_convergence(algo_fun.dejong1, bounds_dj1_D5, "First Dejong Function - Random Search Algorithm - D5", "1. RS_FistDejongConvergenceD5", max_iter, global_dj1_D5, "random_search")
#algo_fun.plot_convergence(algo_fun.dejong2, bounds_dj2_D5, "Second Dejong Function - Random Search Algorithm - D5", "1. RS_SecondDejongConvergenceD5", max_iter, global_dj2_D5, "random_search")
#algo_fun.plot_convergence(algo_fun.schwefel, bounds_swf_D5, "Schwefel Function - Random Search Algorithm - D5", "1. RS_SchwefelConvergenceD5", max_iter, global_swf_D5, "random_search")

#algo_fun.plot_convergence(algo_fun.dejong1, bounds_dj1_D10, "First Dejong Function - Random Search Algorithm - D10", "1. RS_FistDejongConvergenceD10", max_iter, global_dj1_D10, "random_search")
#algo_fun.plot_convergence(algo_fun.dejong2, bounds_dj2_D10, "Second Dejong Function - Random Search Algorithm - D10", "1. RS_SecondDejongConvergenceD10", max_iter, global_dj2_D10, "random_search")
#algo_fun.plot_convergence(algo_fun.schwefel, bounds_swf_D10, "Schwefel Function - Random Search Algorithm - D10", "1. RS_SchwefelConvergenceD10", max_iter, global_swf_D10, "random_search")

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


# define Random Search algorithm
def random_search(objective_func, dim, n_iter, bounds):
    best_solution = None
    best_fitness = float('inf')
    fitness_progress = []

    for i in range(n_iter):

        # Vygenerování náhodného řešení v daných hranicích
        candidate_solution = np.random.uniform(bounds[0], bounds[1], dim)
        candidate_fitness = objective_func(candidate_solution)

        # Porovnání fitness s nejlepším řešením
        if candidate_fitness < best_fitness:
            best_solution = candidate_solution
            best_fitness = candidate_fitness

        fitness_progress.append(best_fitness)

        # Výpis nejlepšího řešení v každé tisícáté iteraci
        #if i % 1000 == 0:

        # print(f'Iteration {i}: Best fitness = {best_fitness}')
    return best_solution, best_fitness, fitness_progress

    num_runs = 30

    # plot fitness progress
    print(f"Iteration {i}: Best solution found: {best_solution}, fitness: {best_fitness}")
    plt.plot(fitness_progress)
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.title('Random Search')
    #plt.show()

    # save plot as png
    plt.savefig(f"C:/Users/vnovotny/Desktop/Garbage/{func}.png")
    plt.clf()
    return best_solution, best_fitness


# test the algorithm with dejong1 function in 5-dimensional space
func = 'dejong1-5'
best_solution, best_fitness = random_search(dejong1, 5, 10000, (-5, 5))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

# test the algorithm with dejong2 function in 5-dimensional space
func = 'dejong2-5'
best_solution, best_fitness = random_search(dejong2, 5, 10000, (-5, 5))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

# test the algorithm with schwefel function in 5-dimensional space
func = 'schwefel-5'
best_solution, best_fitness = random_search(schwefel, 5, 10000, (-500, 500))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

# test the algorithm with dejong1 function in 10-dimensional space
func = 'dejong1-10'
best_solution, best_fitness = random_search(dejong1, 10, 10000, (-5, 5))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

# test the algorithm with dejong2 function in 10-dimensional space
func = 'dejong2-10'
best_solution, best_fitness = random_search(dejong2, 10, 10000, (-5, 5))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

# test the algorithm with schwefel function in 10-dimensional space
func = 'schwefel-10'
best_solution, best_fitness = random_search(schwefel, 10, 10000, (-500, 500))
print(f'Best solution: {best_solution}\nBest fitness: {best_fitness}')

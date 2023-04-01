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
    best_fitnesses = []

    for i in range(n_iter):
        # generate a random solution
        candidate_solution = np.random.uniform(bounds[0], bounds[1], dim)
        candidate_fitness = objective_func(candidate_solution)

        # compare fitness with best solution
        if candidate_fitness < best_fitness:
            best_solution = candidate_solution
            best_fitness = candidate_fitness

        fitness_progress.append(best_fitness)

        # update best fitness from all runs
        #best_fitnesses.append(best_fitness)
        #best_fitness_all_runs = min(best_fitnesses)

        # plot the current best fitness for all runs
        #plt.plot(i, best_fitness_all_runs, marker='x', color='black')

    # return the best solution and fitness value
    return best_solution, best_fitness, fitness_progress

# set the parameters
dims = [5, 5, 5, 10, 10, 10]
functions = [dejong1, dejong2, schwefel, dejong1, dejong2, schwefel]
bounds = [(-5, 5), (-5, 5), (-500, 500), (-5, 5), (-5, 5), (-500, 500)]
n_iter = 10000
num_runs = 30

# run the algorithm for each function and dimension for num_runs times
all_best_fitness = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']
for i, (func, dim, bound) in enumerate(zip(functions, dims, bounds)):
    best_fitnesses = []
    for j in range(num_runs):
        best_solution, best_fitness, fitness_progress = random_search(func, dim, n_iter, bound)
        best_fitnesses.append(best_fitness)
        # plot the fitness progress for each run
        plt.plot(fitness_progress, color=colors[j%len(colors)], alpha=0.5)

    # plot the best fitness value for each function and dimension
    all_best_fitness.append(min(best_fitnesses))
    plt.plot([0, n_iter], [min(best_fitnesses), min(best_fitnesses)], label='Best', color='red')

    # plot the Mean fitness value for each function and dimension
    #all_best_fitness.append(np.mean(best_fitnesses))
    #plt.plot([0, n_iter], [np.mean(best_fitnesses), np.mean(best_fitnesses)], label='Mean', color='black')

    # calculate statistics from the best fitness values of each run
    # print("Minimum:", np.min(all_best_fitness))
    print(f"Minimum: {func.__name__}-{dim} : {np.min(all_best_fitness)}")
    print(f"Maximum: {func.__name__}-{dim} : {np.max(all_best_fitness)}")
    print(f"Mean: {func.__name__}-{dim} : {np.mean(all_best_fitness)}")
    print(f"Median: {func.__name__}-{dim} : {np.median(all_best_fitness)}")
    print(f"Standard deviation: {func.__name__}-{dim} : {np.std(all_best_fitness)}\n\n")

    # set the plot title and axis labels
    plt.title(f'{func.__name__}-{dim}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend()

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Desktop/Garbage/{func.__name__}-{dim}.png')
    plt.clf()

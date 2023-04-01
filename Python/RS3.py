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
        # generate a random solution
        candidate_solution = np.random.uniform(bounds[0], bounds[1], dim)
        candidate_fitness = objective_func(candidate_solution)

        # compare fitness with best solution
        if candidate_fitness < best_fitness:
            best_solution = candidate_solution
            best_fitness = candidate_fitness

        fitness_progress.append(best_fitness)


    # return the best solution and fitness value
    return best_solution, best_fitness, fitness_progress


# set the parameters
dims = [5, 5, 5, 10, 10, 10]
# dims = [5, 10]
functions = [dejong1, dejong2, schwefel, dejong1, dejong2, schwefel]
# functions = [dejong1, dejong1]
bounds = [(-5, 5), (-5, 5), (-500, 500), (-5, 5), (-5, 5), (-500, 500)]
# bounds = [(-5, 5), (-5, 5)]
# n_iter = 5
n_iter = 10000
num_runs = 30

# run the algorithm for each function and dimension for num_runs times
all_best_fitness = []
all_best_fitness_for_run = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']

# iterate over each function, dimension and bound
for i, (func, dim, bound) in enumerate(zip(functions, dims, bounds)):
    best_fitnesses = []
    best_fitness_for_run = []  # inicializace prázdného pole
    # iterate over each run
    for j in range(num_runs):
        best_solution, best_fitness, fitness_progress = random_search(func, dim, n_iter, bound)
        best_fitnesses.append(best_fitness)
        best_fitness_for_run.append(fitness_progress)  # přidání hodnoty do pole
        # plot the fitness progress for each run
        plt.plot(fitness_progress, color=colors[j%len(colors)], alpha=0.5)

    # plot the best fitness value for each function and dimension
    all_best_fitness.append(min(best_fitnesses))

    # calculate statistics from the best fitness values of each run
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
    plt.savefig(f'C:/Users/vnovotny/Desktop/Garbage/{func.__name__}-{dim}-fitness-progress.png')
    plt.clf()

    plt.figure()

    mean_per_column = np.mean(best_fitness_for_run, axis=0)
    print(mean_per_column)
    plt.plot(np.arange(n_iter), mean_per_column, marker='x', markersize=1, label='Mean', color='green')

    min_per_column = np.min(best_fitness_for_run, axis=0)
    print(min_per_column)
    plt.plot(np.arange(n_iter), min_per_column, marker='o', markersize=1, label='Min', color='black')

    # set the plot title and axis labels
    plt.title(f'{func.__name__}-{dim}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend()

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Desktop/Garbage/second_plot_{func.__name__}-{dim}-min-mean.png')

    # clear the current figure
    plt.clf()

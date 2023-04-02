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
    best_solution_rs = None
    best_fitness_rs = float('inf')
    fitness_progress_rs = []

    for i in range(n_iter):
        # generate a random solution
        candidate_solution_rs = np.random.uniform(bounds[0], bounds[1], dim)
        candidate_fitness_rs = objective_func(candidate_solution_rs)

        # compare fitness with best solution
        if candidate_fitness_rs < best_fitness_rs:
            best_solution_rs = candidate_solution_rs
            best_fitness_rs = candidate_fitness_rs

        fitness_progress_rs.append(best_fitness_rs)


    # return the best solution and fitness value
    return best_solution_rs, best_fitness_rs, fitness_progress_rs


# define Simulated Annealing algorithm
def simulated_annealing(objective_func, dim, n_iter, bounds, initial_temperature, final_temperature, cooling_rate):
    current_solution_sa = np.random.uniform(bounds[0], bounds[1], dim)
    current_fitness_sa = objective_func(current_solution_sa)
    best_solution_sa = current_solution_sa
    best_fitness_sa = current_fitness_sa
    fitness_progress_sa = [best_fitness_sa]

    for i in range(n_iter-1):
        # adjust the temperature
        temperature = initial_temperature * ((final_temperature/initial_temperature) ** (i/n_iter))

        # generate a candidate solution
        candidate_solution_sa = current_solution_sa + np.random.normal(0, 1, dim)
        candidate_solution_sa = np.clip(candidate_solution_sa, bounds[0], bounds[1])
        candidate_fitness_sa = objective_func(candidate_solution_sa)

        # check if the candidate solution is better than the current solution
        if candidate_fitness_sa < current_fitness_sa:
            current_solution_sa = candidate_solution_sa
            current_fitness_sa = candidate_fitness_sa

            # check if the candidate solution is better than the best solution
            if candidate_fitness_sa < best_fitness_sa:
                best_solution_sa = candidate_solution_sa
                best_fitness_sa = candidate_fitness_sa
        else:
            # calculate the acceptance probability
            acceptance_prob = np.exp(-(candidate_fitness_sa - current_fitness_sa) / temperature)

            # determine whether to accept the candidate solution
            if np.random.random() < acceptance_prob:
                current_solution_sa = candidate_solution_sa
                current_fitness_sa = candidate_fitness_sa

        fitness_progress_sa.append(best_fitness_sa)
# return the best solution and fitness value
    return best_solution_sa, best_fitness_sa, fitness_progress_sa


# set the parameters
dims = [5, 5, 5, 10, 10, 10]
# dims = [5, 10]
functions = [dejong1, dejong2, schwefel, dejong1, dejong2, schwefel]
# functions = [dejong1, dejong1]
bounds = [(-5, 5), (-5, 5), (-500, 500), (-5, 5), (-5, 5), (-500, 500)]
# bounds = [(-5, 5), (-5, 5)]
n_iter = 10000
# num_runs = 30
# n_iter = 5
num_runs = 30
initial_temperature = 100
final_temperature = 0.01
cooling_rate = 0.92

# run the algorithm for each function and dimension for num_runs times
all_best_fitness_rs = []
all_best_fitness_for_run_rs = []
all_best_fitness_sa = []
all_best_fitness_for_run_sa = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']

# iterate over each function, dimension and bound
for i, (func, dim, bound) in enumerate(zip(functions, dims, bounds)):
    best_fitnesses_rs = []
    best_fitness_for_run_rs = []  # inicializace prázdného pole
    best_fitnesses_sa = []
    best_fitness_for_run_sa = []  # inicializace prázdného pole
    # iterate over each run
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for j in range(num_runs):
        best_solution_rs, best_fitness_rs, fitness_progress_rs = random_search(func, dim, n_iter, bound)
        best_solution_sa, best_fitness_sa, fitness_progress_sa = simulated_annealing(func, dim, n_iter, bound, initial_temperature, final_temperature, cooling_rate)
        best_fitnesses_rs.append(best_fitness_rs)
        best_fitnesses_sa.append(best_fitness_sa)
        best_fitness_for_run_rs.append(fitness_progress_rs)  # přidání hodnoty do pole
        best_fitness_for_run_sa.append(fitness_progress_sa)  # přidání hodnoty do pole
        # plot the fitness progress for each run
        #plt.plot(fitness_progress_rs, color=colors[j % len(colors)], alpha=0.5)
        #plt.plot(fitness_progress_sa, color=colors[j % len(colors)], alpha=0.5)
        ax1.plot(fitness_progress_rs, color=colors[j % len(colors)], alpha=0.5)
        ax2.plot(fitness_progress_sa, color=colors[j % len(colors)], alpha=0.5)

    # plot the best fitness value for each function and dimension
    all_best_fitness_rs.append(min(best_fitnesses_rs))
    all_best_fitness_sa.append(min(best_fitnesses_sa))

    # calculate statistics from the best fitness values of each run
    # print(f"Minimum: {func.__name__}-{dim} : {np.min(all_best_fitness_rs)}")
    # print(f"Maximum: {func.__name__}-{dim} : {np.max(all_best_fitness_rs)}")
    # print(f"Mean: {func.__name__}-{dim} : {np.mean(all_best_fitness_rs)}")
    # print(f"Median: {func.__name__}-{dim} : {np.median(all_best_fitness_rs)}")
    # print(f"Standard deviation: {func.__name__}-{dim} : {np.std(all_best_fitness_rs)}\n\n")

    # set the plot title and axis labels
    #plt.title(f'{func.__name__}-{dim}')
    #plt.xlabel('Number of iterations')
    #plt.ylabel('Fitness')
    # plt.legend()

    # save the plot
    #plt.savefig(f'C:/Users/vnovotny/Desktop/RS-SA-TEST/{func.__name__}-{dim}-fitness-progress.png')
    #plt.clf()

    ax1.set_title(f'{func.__name__}-{dim} Random Search')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Fitness')

    # set the title and axis labels for the second subplot
    ax2.set_title(f'{func.__name__}-{dim} Simulated Annealing')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Fitness')

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Desktop/RS-SA-TEST/{func.__name__}-{dim}-fitness-progress.png')
    plt.clf()

    plt.figure()

    mean_per_column_rs = np.mean(best_fitness_for_run_rs, axis=0)
    print(mean_per_column_rs)
    plt.plot(np.arange(n_iter), mean_per_column_rs, marker='x', markersize=1, label='Mean-RS', color='green')

    min_per_column_rs = np.min(best_fitness_for_run_rs, axis=0)
    print(min_per_column_rs)
    plt.plot(np.arange(n_iter), min_per_column_rs, marker='o', markersize=1, label='Min-RS', color='black')

    mean_per_column_sa = np.mean(best_fitness_for_run_sa, axis=0)
    # print(mean_per_column_sa)
    plt.plot(np.arange(n_iter), mean_per_column_sa, marker='x', markersize=1, label='Mean-SA', color='red')

    min_per_column_sa = np.min(best_fitness_for_run_sa, axis=0)
    # print(min_per_column_sa)
    plt.plot(np.arange(n_iter), min_per_column_sa, marker='o', markersize=1, label='Min-SA', color='gray')

    # set the plot title and axis labels
    plt.title(f'{func.__name__}-{dim}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend()

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Desktop/RS-SA-TEST/{func.__name__}-{dim}-min-mean.png')

    # clear the current figure
    plt.clf()

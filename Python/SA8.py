import numpy as np
import matplotlib.pyplot as plt
import time


# Vydefinovani funkce 1st Dejong pro x dimenzionalni prostor
def dejong1(x):
    return sum([xi ** 2 for xi in x])


# Vydefinovani funkce 2nd Dejong pro x dimenzionalni prostor
def dejong2(x):
    assert len(x) >= 2
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# Vydefinovani schwefel funkce pro x dimenzionalni prostor
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


# Random Search
def random_search(objective_func, dimens, n_iters, f_bounds):
    # Inicializace: Nastaveni best_solutions na prazdne, best_fitness na nejvetsi moznou hodnotu
    best_solution_rs_f = None
    best_fitness_rs_f = float('inf')
    fitness_progress_rs_f = []

    for i_f in range(n_iters):
        # vygenerovani nahodneho kandidata
        candidate_solution_rs = np.random.uniform(f_bounds[0], f_bounds[1], dimens)
        candidate_fitness_rs = objective_func(candidate_solution_rs)

        # porovnani fitness kandidata a doposud nejlepsiho reseni
        if candidate_fitness_rs < best_fitness_rs_f:
            best_solution_rs_f = candidate_solution_rs
            best_fitness_rs_f = candidate_fitness_rs

        # pridani best_fitness do pole fitness_progress
        fitness_progress_rs_f.append(best_fitness_rs_f)

    # funkce vraci tyto hodnoty, pole
    return best_solution_rs_f, best_fitness_rs_f, fitness_progress_rs_f


# define Simulated Annealing algorithm
def simulated_annealing(objective_func, dimensions, n_iter_f, bounds_f, initial_temperature_f, final_temperature_f, cooling_rate_f):
    # vygenerovani nahodneho kandidata
    current_solution_sa = np.random.uniform(bounds_f[0], bounds_f[1], dimensions)
    current_fitness_sa = objective_func(current_solution_sa)
    # Inicializace: z prave vygenerovanych hodnot udela nejlepsi a algoritmus muze zacit,
    # akorat pro n -1, protoze prvni prvek uz je vygenerovan
    best_solution_sa_f = current_solution_sa
    best_fitness_sa_f = current_fitness_sa
    fitness_progress_sa_f = [best_fitness_sa_f]
    # -0,1 až 0,1
    #okoli = -0.1*(bounds_f[1]-bounds_f[0]), 0.1*(bounds_f[1]-bounds_f[0]), dimensions
    okoli = (bounds_f[1]-bounds_f[0])

    for i_f in range(n_iter_f-1):
        # exponencialni ochlazovani
        # temperature = initial_temperature * ((final_temperature_f/initial_temperature_f) ** (i/n_iter))
        temperature_f = initial_temperature_f * ((final_temperature_f/initial_temperature_f) ** (i/n_iter_f))

        # vygenerovani nahodneho kandidata - Nejprve se vytvori nove kandidatní reseni pridanim nahodného sumu
        candidate_solution_sa = current_solution_sa + np.random.uniform(-0.1*okoli, 0.1*okoli, dimensions)
        # omezeni jej na urcený interval, aby se zajistilo, ze nove reseni bude v souladu s danymi omezenimi.
        candidate_solution_sa = np.clip(candidate_solution_sa, bounds_f[0], bounds_f[1])
        # vypocet fitness hodnoty
        candidate_fitness_sa = objective_func(candidate_solution_sa)

        # kontrola zda je kandidatni reseni lepsi nez aktualni reseni
        if candidate_fitness_sa < current_fitness_sa:
            current_solution_sa = candidate_solution_sa
            current_fitness_sa = candidate_fitness_sa

            # kontrola zda je kandidatni reseni lepsi nez nejlepsi reseni
            if candidate_fitness_sa < best_fitness_sa_f:
                best_solution_sa_f = candidate_solution_sa
                best_fitness_sa_f = candidate_fitness_sa
        else:
            # vypocet akceptovani zhorseneho vyyledku
            acceptance_prob = np.exp(-(candidate_fitness_sa - current_fitness_sa) / temperature_f)

            # rozhodnuti zda akceptovat zhorseni vysledku
            if np.random.random() < acceptance_prob:
                current_solution_sa = candidate_solution_sa
                current_fitness_sa = candidate_fitness_sa

        fitness_progress_sa_f.append(best_fitness_sa_f)
    # funkce vraci tyto hodnoty, pole
    return best_solution_sa_f, best_fitness_sa_f, fitness_progress_sa_f


# set the parameters
dims = [5, 5, 5, 10, 10, 10]
# dims = [5, 10]
functions = [dejong1, dejong2, schwefel, dejong1, dejong2, schwefel]
# functions = [dejong1, dejong1]
bounds = [(-5, 5), (-5, 5), (-500, 500), (-5, 5), (-5, 5), (-500, 500)]
# bounds = [(-5, 5), (-5, 5)]
n_iter_rs = 10000
metropolis_calls = 10
n_iter_sa = int(n_iter_rs/metropolis_calls)
# num_runs = 30
# n_iter = 5
num_runs = 30
temperature = 100
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
    start_time = time.time()  # Get the current time
    # iterate over each run
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for j in range(num_runs):
        best_solution_rs, best_fitness_rs, fitness_progress_rs = random_search(func, dim, n_iter_rs, bound)
        best_solution_sa, best_fitness_sa, fitness_progress_sa = simulated_annealing(func, dim, n_iter_rs, bound, initial_temperature, final_temperature, cooling_rate)
        best_fitnesses_rs.append(best_fitness_rs)
        best_fitnesses_sa.append(best_fitness_sa)
        best_fitness_for_run_rs.append(fitness_progress_rs)  # přidání hodnoty do pole
        best_fitness_for_run_sa.append(fitness_progress_sa)  # přidání hodnoty do pole
        # plot the fitness progress for each run
        # plt.plot(fitness_progress_rs, color=colors[j % len(colors)], alpha=0.5)
        # plt.plot(fitness_progress_sa, color=colors[j % len(colors)], alpha=0.5)
        ax1.plot(fitness_progress_rs, color=colors[j % len(colors)], alpha=0.5)
        ax2.plot(fitness_progress_sa, color=colors[j % len(colors)], alpha=0.5)

    # plot the best fitness value for each function and dimension
    end_time = time.time()  # Get the current time again
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("Elapsed time:" + str(elapsed_time) + "seconds\n")

    all_best_fitness_rs.append(min(best_fitnesses_rs))
    all_best_fitness_sa.append(min(best_fitnesses_sa))

    # calculate statistics from the best fitness values of each run
    # print(f"Minimum: {func.__name__}-{dim} : {np.min(all_best_fitness_rs)}")
    # print(f"Maximum: {func.__name__}-{dim} : {np.max(all_best_fitness_rs)}")
    # print(f"Mean: {func.__name__}-{dim} : {np.mean(all_best_fitness_rs)}")
    # print(f"Median: {func.__name__}-{dim} : {np.median(all_best_fitness_rs)}")
    # print(f"Standard deviation: {func.__name__}-{dim} : {np.std(all_best_fitness_rs)}\n\n")
    # print(f"Cooling rate: {((final_temperature / initial_temperature) ** (1 / n_iter))}\n")

    # set the plot title and axis labels
    # plt.title(f'{func.__name__}-{dim}')
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Fitness')
    # plt.legend()

    # save the plot
    # plt.savefig(f'C:/Users/vnovotny/Desktop/RS-SA-TEST/{func.__name__}-{dim}-fitness-progress.png')
    # plt.clf()

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
    # print(mean_per_column_rs)
    plt.plot(np.arange(n_iter_rs), mean_per_column_rs, marker='x', markersize=1, label='Mean-RS', color='green')

    min_per_column_rs = np.min(best_fitness_for_run_rs, axis=0)
    # print(min_per_column_rs)
    plt.plot(np.arange(n_iter_rs), min_per_column_rs, marker='o', markersize=1, label='Min-RS', color='black')

    mean_per_column_sa = np.mean(best_fitness_for_run_sa, axis=0)
    # print(mean_per_column_sa)
    plt.plot(np.arange(n_iter_rs), mean_per_column_sa, marker='x', markersize=1, label='Mean-SA', color='red')

    min_per_column_sa = np.min(best_fitness_for_run_sa, axis=0)
    # print(min_per_column_sa)
    plt.plot(np.arange(n_iter_rs), min_per_column_sa, marker='o', markersize=1, label='Min-SA', color='gray')

    # set the plot title and axis labels
    plt.title(f'{func.__name__}-{dim}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend()

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Desktop/RS-SA-TEST/{func.__name__}-{dim}-min-mean.png')

    # clear the current figure
    plt.clf()

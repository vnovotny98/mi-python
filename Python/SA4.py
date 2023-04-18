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
def simulated_annealing(objective_func, dimensions, n_iter_f, bounds_f, temperature_f, cooling_rate_f):
    # vygenerovani nahodneho kandidata
    current_solution_sa = np.random.uniform(bounds_f[0], bounds_f[1], dimensions)
    current_fitness_sa = objective_func(current_solution_sa)
    # Inicializace: z prave vygenerovane hodnoty udela tu nejlepsi a algoritmus muze zacit,
    best_solution_sa_f = current_solution_sa
    best_fitness_sa_f = current_fitness_sa
    fitness_progress_sa_f = [best_fitness_sa_f]
    okoli = (bounds_f[1]-bounds_f[0])
    for i_f in range(n_iter_f):
        # exponencialni ochlazovani
        temperature_f *= cooling_rate_f
        # Loop through 10 candidates
        for j_f in range(metropolis_calls):
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

                    # Update the best fitness and solution if necessary
                    # (in case a better solution was found in the inner loop)
                    current_best_fitness = current_fitness_sa
                    if current_best_fitness < best_fitness_sa_f:
                        best_fitness_sa_f = current_best_fitness
                        best_solution_sa_f = current_solution_sa

            fitness_progress_sa_f.append(best_fitness_sa_f)
        # funkce vraci tyto hodnoty, pole
    return best_solution_sa_f, best_fitness_sa_f, fitness_progress_sa_f


# set the parameters
# dims = [5, 10]
# functions = [dejong1, dejong1]
# bounds = [(-5, 5), (-5, 5)]
dims = [5, 5, 5, 10, 10, 10]
functions = [dejong1, dejong2, schwefel, dejong1, dejong2, schwefel]
bounds = [(-5, 5), (-5, 5), (-500, 500), (-5, 5), (-5, 5), (-500, 500)]

n_iter_rs = 10000
num_runs = 30
temperature = 700
cooling_rate = 0.98
metropolis_calls = 50
n_iter_sa = int(n_iter_rs/metropolis_calls)


# run the algorithm for each function and dimension for num_runs times
all_best_fitness_rs = []
all_best_fitness_for_run_rs = []
all_best_fitness_sa = []
all_best_fitness_for_run_sa = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']

# iterate over each function, dimension and bound
for i, (func, dim, bound) in enumerate(zip(functions, dims, bounds)):
    best_fitnesses_rs = []
    best_solution_rs_all = []
    best_fitness_for_run_rs = []  # inicializace prázdného pole
    best_fitnesses_sa = []
    best_solution_sa_all = []
    best_fitness_for_run_sa = []  # inicializace prázdného pole
    start_time = time.time()  # Get the current time
    # iterate over each run
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for j in range(num_runs):
        best_solution_rs, best_fitness_rs, fitness_progress_rs = random_search(func, dim, n_iter_rs, bound)
        best_solution_sa, best_fitness_sa, fitness_progress_sa = \
            simulated_annealing(func, dim, n_iter_sa, bound, temperature, cooling_rate)
        best_solution_rs_all.append(best_solution_rs)
        best_solution_sa_all.append(best_solution_sa)
        best_fitnesses_rs.append(best_fitness_rs)
        best_fitnesses_sa.append(best_fitness_sa)
        best_fitness_for_run_rs.append(fitness_progress_rs)  # přidání hodnoty do pole
        best_fitness_for_run_sa.append(fitness_progress_sa)  # přidání hodnoty do pole
        # plot the fitness progress for each run
        ax1.plot(fitness_progress_rs, color=colors[j % len(colors)], alpha=0.5)
        ax2.plot(fitness_progress_sa, color=colors[j % len(colors)], alpha=0.5)

    # Find the index of the run with the lowest best_fitness_sa
    min_fitness_idx_sa = best_fitnesses_sa.index(min(best_fitnesses_sa))
    min_fitness_idx_rs = best_fitnesses_rs.index(min(best_fitnesses_rs))
    # Print the best_solution_rs and best_solution_sa with the lowest best_fitness_sa value
    print(f"Best solution RS {func.__name__}-{dim}: {best_solution_rs_all[min_fitness_idx_rs]}")
    print(f"Best fitness RS {func.__name__}-{dim}: {best_fitnesses_rs[min_fitness_idx_rs]}")
    print(f"Best solution SA {func.__name__}-{dim}: {best_solution_sa_all[min_fitness_idx_rs]}")
    print(f"Best fitness SA {func.__name__}-{dim}: {best_fitnesses_sa[min_fitness_idx_sa]}")

    end_time = time.time()  # Get the current time again
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print("Elapsed time:" + str(elapsed_time) + "seconds\n")

    # plot the best fitness value for each function and dimension
    all_best_fitness_rs.append(min(best_fitnesses_rs))
    all_best_fitness_sa.append(min(best_fitnesses_sa))

    # calculate statistics from the best fitness values of each run
    print(f"Minimum RS: {func.__name__}-{dim} : {np.min(best_fitness_for_run_rs)}")
    print(f"Maximum RS: {func.__name__}-{dim} : {np.max(best_fitness_for_run_rs)}")
    print(f"Mean RS: {func.__name__}-{dim} : {np.mean(best_fitness_for_run_rs)}")
    print(f"Median RS: {func.__name__}-{dim} : {np.median(best_fitness_for_run_rs)}")
    print(f"Standard deviation RS: {func.__name__}-{dim} : {np.std(best_fitness_for_run_rs)}\n\n")

    print(f"Minimum SA: {func.__name__}-{dim} : {np.min(best_fitness_for_run_sa)}")
    print(f"Maximum SA: {func.__name__}-{dim} : {np.max(best_fitness_for_run_sa)}")
    print(f"Mean SA: {func.__name__}-{dim} : {np.mean(best_fitness_for_run_sa)}")
    print(f"Median SA: {func.__name__}-{dim} : {np.median(best_fitness_for_run_sa)}")
    print(f"Standard deviation SA: {func.__name__}-{dim} : {np.std(best_fitness_for_run_sa)}\n\n")
    print(f"Final temperature SA: {(temperature * (cooling_rate ** n_iter_sa))}\n")

    all_best_fitness_rs.clear()
    all_best_fitness_sa.clear()

    # set the plot title and axis labels
    ax1.set_title(f'{func.__name__}-{dim} Random Search')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Fitness')

    # set the title and axis labels for the second subplot
    ax2.set_title(f'{func.__name__}-{dim} Simulated Annealing')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Fitness')

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Documents/Matematická informatika/Benchmarking/{func.__name__}-{dim}-fitness'
                f'-progress.png')
    plt.clf()

    plt.figure()

    mean_per_column_rs = np.mean(best_fitness_for_run_rs, axis=0)
    plt.plot(np.arange(n_iter_rs), mean_per_column_rs, marker='x', markersize=1, label='Mean-RS', color='green')

    min_per_column_rs = np.min(best_fitness_for_run_rs, axis=0)
    plt.plot(np.arange(n_iter_rs), min_per_column_rs, marker='o', markersize=1, label='Min-RS', color='black')

    mean_per_column_sa = np.mean(best_fitness_for_run_sa, axis=0)
    plt.plot(np.arange(n_iter_rs+1), mean_per_column_sa, marker='x', markersize=1, label='Mean-SA', color='red')
    plt.plot(np.arange(n_iter_rs), mean_per_column_sa, marker='x', markersize=1, label='Mean-SA', color='red')
    min_per_column_sa = np.min(best_fitness_for_run_sa, axis=0)
    plt.plot(np.arange(n_iter_rs+1), min_per_column_sa, marker='o', markersize=1, label='Min-SA', color='gray')
    plt.plot(np.arange(n_iter_rs), min_per_column_sa, marker='o', markersize=1, label='Min-SA', color='gray')

    # set the plot title and axis labels
    plt.title(f'{func.__name__}-{dim}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.legend()

    # save the plot
    plt.savefig(f'C:/Users/vnovotny/Documents/Matematická informatika/Benchmarking/{func.__name__}-{dim}-min-mean.png')

    # clear the current figure
    plt.clf()

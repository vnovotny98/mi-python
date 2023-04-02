import random
import math

# funkce 1st Dejong
def dejong(x):
    return sum([i**2 for i in x])

# funkce pro výpočet nové teploty
def temperature(curr_temp, alpha):
    return curr_temp * alpha

# funkce pro výpočet pravděpodobnosti přijetí nového řešení
def acceptance_probability(curr_cost, new_cost, temp):
    if new_cost < curr_cost:
        return 1.0
    else:
        return math.exp((curr_cost - new_cost) / temp)

# simulované žíhání
def simulated_annealing(num_runs, M, x0, N, f, T0, Tf, alpha, nT):
    best_results = []
    for run in range(num_runs):
        curr_solution = x0
        curr_cost = f(curr_solution)

        best_solution = curr_solution
        best_cost = curr_cost

        curr_temp = T0
        while curr_temp > Tf:
            for i in range(nT):
                # vyber souseda
                rand_neighbor = random.choice(N(curr_solution))
                # výpočet nového řešení
                new_solution = M(curr_solution, rand_neighbor)
                new_cost = f(new_solution)
                # výpočet pravděpodobnosti přijetí nového řešení
                ap = acceptance_probability(curr_cost, new_cost, curr_temp)
                # rozhodnutí, zda nové řešení přijmout nebo ne
                if ap > random.random():
                    curr_solution = new_solution
                    curr_cost = new_cost
                    # uložení nejlepšího řešení
                    if curr_cost < best_cost:
                        best_solution = curr_solution
                        best_cost = curr_cost
            # snížení teploty
            curr_temp = temperature(curr_temp, alpha)
        # uložení nejlepšího výsledku pro tento běh
        best_results.append(best_cost)
    return best_results


# inicializace
num_runs = 30
M = lambda x, neighbor: [x[i] + random.uniform(-1, 1) for i in range(len(x))]
x0 = [random.uniform(-5.12, 5.12) for i in range(5)]
N = lambda x: [(x[i] + delta) if random.random() < 0.5 else (x[i] - delta) for i in range(len(x)) for delta in [-0.1, 0, 0.1]]
f = dejong
T0 = 100.0
Tf = 1e-8
alpha = 0.99
nT = len(N(x0))

# spuštění algoritmu a vypsání nejlepších výsledků
best_results = simulated_annealing(num_runs, M, x0, N, f, T0, Tf, alpha, nT)
print("Nejlepší výsledky:", best_results)
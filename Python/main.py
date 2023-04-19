from KS_instance import generate_knapsack_instance
from KS_bruteforce import knapsack_brute_force
from KS_simulovanezihani import knapsack_simulated_annealing

# volání funkce pro generování instance problému batohu
values, weights, capacity = generate_knapsack_instance()
values_sa = values
weights_sa = weights
capacity_sa = capacity

# volání algoritmu
max_value_rs, best_weight_rs, max_combination_rs, total_time_rs, num_iterations_bf = \
    knapsack_brute_force(values, weights, capacity)

# výpis výsledků
print("Maximální hodnota pomocí Brute Force:", max_value_rs)
print("Pouzita kapacita pomocí Brute Force:", best_weight_rs)
print("Kombinace předmětů pomocí Brute Force:", max_combination_rs)
print("Celkový čas potřebný pro Brute Force:", total_time_rs)
print("Počet iterací Brute Force:", num_iterations_bf)

max_iterations = 10000
num_runs = 30
temperature = 250
cooling_rate = 0.98
metropolis_calls = 20
n_iter_sa = int(max_iterations / metropolis_calls)

# volání algoritmu Simulated Annealing
best_value, best_weight, max_combination, total_time, num_iterations_sa = \
    knapsack_simulated_annealing(values_sa, weights_sa, capacity_sa, max_iterations, temperature, cooling_rate)

# výpis výsledků
print("\nMaximální hodnota pomocí Simulated Annealing:", best_value)
print("Pouzita kapacita pomocí Simulated Annealing:", best_weight)
print("Kombinace předmětů pomocí Simulated Annealing:",
      [i + 1 for i in range(len(max_combination)) if max_combination[i] == 1])
# print("Kombinace předmětů pomocí Simulated Annealing:", max_combination)
print("Celkový čas potřebný pro Simulated Annealing:", total_time)
print("Počet iterací Simulated Annealing:", num_iterations_sa, "\n")

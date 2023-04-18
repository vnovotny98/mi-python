from KS_instance import generate_knapsack_instance
from KS_bruteforce import knapsack_brute_force
from KS_simulovanezihani import knapsack_simulated_annealing

# volání funkce pro generování instance problému batohu
values, weights, capacity = generate_knapsack_instance()
values_sa = values
weights_sa = weights
capacity_sa = capacity

# volání algoritmu
max_value_rs, best_weight_rs, max_combination_rs, total_time_rs = knapsack_brute_force(values, weights, capacity)

# výpis výsledků
print("Maximální hodnota pomocí Brute Force:", max_value_rs)
print("Pouzita kapacita pomocí Brute Force:", best_weight_rs)
print("Kombinace předmětů pomocí Brute Force:", max_combination_rs)
print("Celkový čas potřebný pro Brute Force:", total_time_rs)

max_iterations = 10000
initial_temperature = 100
cooling_factor = 0.95

# volání algoritmu Simulated Annealing
best_value, best_weight, max_combination, total_time =\
    knapsack_simulated_annealing(values_sa, weights_sa, capacity_sa, max_iterations, initial_temperature, cooling_factor)

# výpis výsledků
print("Maximální hodnota pomocí Simulated Annealing:", best_value)
print("Pouzita kapacita pomocí Simulated Annealing:", best_weight)
print("Kombinace předmětů pomocí Simulated Annealing:",
      [i+1 for i in range(len(max_combination)) if max_combination[i] == 1])
print("Kombinace předmětů pomocí Simulated Annealing:", max_combination)
print("Celkový čas potřebný pro Simulated Annealing:", total_time)

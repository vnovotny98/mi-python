from KS_instance import generate_knapsack_instance
from KS_bruteforce import knapsack_brute_force

# volání funkce pro generování instance problému batohu
values, weights, capacity = generate_knapsack_instance()

# volání algoritmu
max_value, best_weight, max_combination, total_time = knapsack_brute_force(values, weights, capacity)

# výpis výsledků
print("Maximální hodnota pomocí Brute Force:", max_value)
print("Pouzita kapacita pomocí Brute Force:", best_weight)
print("Kombinace předmětů pomocí Brute Force:", max_combination)
print("Celkový čas potřebný pro Brute Force:", total_time)

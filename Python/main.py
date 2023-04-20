import matplotlib.pyplot as plt

from KS_instance import generate_knapsack_instance
from KS_bruteforce import knapsack_brute_force
from KS_SA_Metropolis import knapsack_simulated_annealing

# volání funkce pro generování instance problému batohu
values, volumes, capacity = generate_knapsack_instance()
values_sa = values
volumes_sa = volumes
capacity_sa = capacity

# volání algoritmu
max_value_rs, best_volume_rs, max_combination_rs, total_time_rs, num_iterations_bf, value_progress_bf = \
    knapsack_brute_force(values, volumes, capacity)

# výpis výsledků
print("Maximální hodnota pomocí Brute Force:", max_value_rs)
print("Pouzita kapacita pomocí Brute Force:", best_volume_rs)
print("Kombinace předmětů pomocí Brute Force:", max_combination_rs)
print("Celkový čas potřebný pro Brute Force:", total_time_rs)
print("Počet iterací Brute Force:", num_iterations_bf)

plt.plot(value_progress_bf)
plt.title('Brute Force - knapsack')
plt.xlabel('Iterace')
plt.ylabel('Nejlepší hodnota')
plt.savefig(f'C:/Users/vnovotny/Documents/Matematická informatika/Knapsack/brute-force-knapsack.png')
plt.clf()

max_iterations = 10000
num_runs = 30
temperature = 250
cooling_rate = 0.98
metropolis_calls = 20
n_iter_sa = int(max_iterations / metropolis_calls)

# inicializace seznamu pro ukládání nejlepších hodnot z 30 běhů
best_values = []
value_progress_sa_runs = []

for i in range(num_runs):
    # volání algoritmu Simulated Annealing
    best_value, best_volume, max_combination, total_time, num_iterations_sa, value_progress_sa = \
        knapsack_simulated_annealing(values_sa, volumes_sa, capacity_sa, max_iterations,
                                     temperature, cooling_rate, metropolis_calls)

    # výpis výsledků pro aktuální běh
    print(f"\nVýsledky běhu {i+1}:")
    print("Maximální hodnota pomocí Simulated Annealing:", best_value)
    print("Pouzita kapacita pomocí Simulated Annealing:", best_volume)
    print("Kombinace předmětů pomocí Simulated Annealing:",
          [j + 1 for j in range(len(max_combination)) if max_combination[j] == 1])
    print("Celkový čas potřebný pro Simulated Annealing:", total_time)
    print("Počet iterací Simulated Annealing:", num_iterations_sa)

    # přidání nejlepší hodnoty z aktuálního běhu do seznamu nejlepších hodnot
    best_values.append(best_value)
    value_progress_sa_runs.append(value_progress_sa)

plt.figure()  # vytvoření nového obrázku
for i in range(num_runs):
    plt.plot(value_progress_sa_runs[i], label=f'Run {i+1}')  # přidání průběhu z aktuálního běhu
plt.title('Simulated Annealing - knapsack')
plt.xlabel('Iterace')
plt.ylabel('Nejlepší hodnota')
plt.savefig(f'C:/Users/vnovotny/Documents/Matematická informatika/Knapsack/simulated-annealing-knapsack.png')
plt.clf()

# výpis nejlepší hodnoty z 30 běhů
print("\nNejlepší hodnoty ze 30 běhů:", best_values)

# vytvoření grafu pro nejlepší hodnoty z 30 běhů
plt.plot(best_values)
plt.title('Simulated Annealing - knapsack - nejlepší hodnoty z 30 běhů')
plt.xlabel('Běh')
plt.ylabel('Nejlepší hodnota')
plt.savefig(f'C:/Users/vnovotny/Documents/Matematická informatika/Knapsack/simulated-annealing-best-values.png')
plt.clf()

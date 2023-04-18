import random
import time


def knapsack_simulated_annealing(values, weights, capacity, max_iterations, initial_temperature, cooling_factor):
    # inicializace aktuálního a nejlepšího řešení
    current_solution = [0] * len(values)
    best_solution = None
    best_value = 0
    best_weight = 0

    # inicializace teploty
    temperature = initial_temperature

    # start stopky
    start_time = time.time()

    # hlavní smyčka
    for i in range(max_iterations):
        # vygenerování nového náhodného řešení
        new_solution = [random.randint(0, 1) for _ in range(len(values))]

        # výpočet hodnoty a váhy nového řešení
        new_value = sum(values[j] for j in range(len(values)) if new_solution[j] == 1)
        new_weight = sum(weights[j] for j in range(len(weights)) if new_solution[j] == 1)

        # pokud nové řešení splňuje kapacitu batohu a má lepší hodnotu, přijmi ho jako aktuální řešení
        if new_weight <= capacity and \
                (new_value > sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
                 or current_solution == [0] * len(values)):
            current_solution = new_solution
        # jinak přijmi ho s pravděpodobností určenou teplotou a rozdílem hodnoty
        else:
            delta = new_value - sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
            if random.random() < pow(2.71828, delta / temperature):
                current_solution = new_solution

        # aktualizace nejlepšího řešení
        if sum(values[j] for j in range(len(values)) if current_solution[j] == 1) > best_value:
            best_solution = current_solution.copy()
            best_value = sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
            best_weight = sum(weights[j] for j in range(len(weights)) if current_solution[j] == 1)

        # snížení teploty
        temperature *= cooling_factor

    # konec stopky
    end_time = time.time()

    # celkový čas
    total_time = end_time - start_time

    # vrácení nejlepšího řešení
    # return best_value, best_weight, [i+1 for i in range(len(best_solution)) if best_solution[i] == 1], total_time

    return best_value, best_weight, best_solution, total_time

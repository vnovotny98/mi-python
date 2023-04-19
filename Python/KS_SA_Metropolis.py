import random
import time
import numpy as np


def knapsack_simulated_annealing(values, weights, capacity, max_iterations,
                                 initial_temperature, cooling_factor, metropolis_calls):
    # inicializace aktuálního a nejlepšího řešení
    current_solution = [0] * len(values)
    best_solution = None
    best_value = 0
    best_weight = 0
    num_iterations = 0

    # inicializace teploty
    temperature = initial_temperature

    # inicializace seznamu pro ukládání nejlepší hodnoty
    value_progress = []

    # start stopky
    start_time = time.time()

    # hlavní smyčka
    for i in range(max_iterations // metropolis_calls):
        # snížení teploty
        temperature *= cooling_factor

        # spustíme Metropolis metodu metropolis_calls krát
        for j in range(metropolis_calls):
            # vygenerování nového řešení v okolí new_solution
            neighbor_solution = [random.randint(0, 1) for _ in range(len(values))]
            index = random.randint(0, len(values) - 1)
            neighbor_solution[index] = 1 - neighbor_solution[index]
            num_iterations += 1
            # výpočet hodnoty a váhy nového řešení
            neighbor_value = sum(values[j] for j in range(len(values)) if neighbor_solution[j] == 1)
            neighbor_weight = sum(weights[j] for j in range(len(weights)) if neighbor_solution[j] == 1)

            # pokud nové řešení splňuje kapacitu batohu a má lepší hodnotu, přijmi ho jako aktuální řešení
            if neighbor_weight <= capacity and \
                    (neighbor_value > sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
                     or current_solution == [0] * len(values)):
                current_solution = neighbor_solution
            # jinak přijmi ho s pravděpodobností určenou teplotou a rozdílem hodnoty
            else:
                delta = neighbor_value - sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
                if neighbor_weight > capacity:
                    current_solution = current_solution
                elif random.random() < np.exp(delta / temperature):
                    current_solution = neighbor_solution

            # aktualizace nejlepšího řešení
            if sum(values[j] for j in range(len(values)) if current_solution[j] == 1) > best_value:
                best_solution = current_solution.copy()
                best_value = sum(values[j] for j in range(len(values)) if current_solution[j] == 1)
                best_weight = sum(weights[j] for j in range(len(weights)) if current_solution[j] == 1)

            # přidání aktuální hodnoty do seznamu průběhu hodnot
            value_progress.append(best_value)

    # konec stopky
    end_time = time.time()

    # celkový čas
    total_time = end_time - start_time

    # vrácení nejlepšího řešení
    return best_value, best_weight, best_solution, total_time, num_iterations, value_progress

import itertools
import time


def knapsack_brute_force(values, weights, capacity):
    # inicializace maximální hodnoty a kombinace předmětů
    max_value = 0
    best_weight = 0
    max_combination = None

    # start stopky
    start_time = time.time()

    # projití všech možných kombinací předmětů
    for i in range(1, len(values) + 1):
        for combination in itertools.combinations(range(len(values)), i):
            # výpočet hodnoty a váhy kombinace předmětů
            combination_value = sum(values[j] for j in combination)
            combination_weight = sum(weights[j] for j in combination)

            # pokud kombinace splňuje kapacitu batohu a má vyšší hodnotu, aktualizuj maximum
            if combination_weight <= capacity and combination_value > max_value:
                max_value = combination_value
                best_weight = combination_weight
                max_combination = combination

    # konec stopky
    end_time = time.time()

    # celkový čas
    total_time = end_time - start_time

    # vrácení maximální hodnoty a kombinace
    return max_value, best_weight, max_combination, total_time

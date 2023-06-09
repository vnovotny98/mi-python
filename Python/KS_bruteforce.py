import itertools
import time


def knapsack_brute_force(values, volumes, capacity):
    # inicializace maximální hodnoty a kombinace předmětů
    max_value = 0
    best_volume = 0
    max_combination = None
    num_iterations = 0

    # start stopky
    start_time = time.time()

    # inicializace seznamu průběhu hodnot
    value_progress = []

    # projití všech možných kombinací předmětů
    for i in range(1, len(values) + 1):
        for combination in itertools.combinations(range(1, len(values) + 1), i):
            # výpočet hodnoty a objemu kombinace předmětů
            combination_value = sum(values[j-1] for j in combination)
            combination_volume = sum(volumes[j-1] for j in combination)

            num_iterations += 1

            # pokud kombinace splňuje kapacitu batohu a má vyšší hodnotu, aktualizuj maximum
            if combination_volume <= capacity and combination_value > max_value:
                max_value = combination_value
                best_volume = combination_volume
                max_combination = combination

            # přidání aktuální hodnoty do seznamu průběhu hodnot
            value_progress.append(max_value)

    # konec stopky
    end_time = time.time()

    # celkový čas
    total_time = end_time - start_time

    # vrácení maximální hodnoty a kombinace
    return max_value, best_volume, max_combination, total_time, num_iterations, value_progress

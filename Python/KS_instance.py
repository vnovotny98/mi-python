import random


def generate_knapsack_instance():
    # nastavení počtu předmětů
    n_items = 20
    # nastavení kapacity batohu - 16 - 30 = 200
    capacity = 200
    # inicializace seznamů pro váhy a hodnoty předmětů
    weights = []
    values = []

    # generování náhodných váh a hodnot předmětů
    for i in range(n_items):
        weights.append(random.randint(1, 50))
        values.append(random.randint(1, 50))

    # výpočet celkové váhy a hodnoty předmětů a výpis včetně jejich ID
    total_weight = 0
    total_value = 0
    for i in range(n_items):
        total_weight += weights[i]
        total_value += values[i]
        print("ID:", i+1, "Objem:", weights[i], "Cena:", values[i])

    # kontrola, zda kapacita batohu není menší než celkový objem předmětů - nemělo by cenu pak pouštět algoritmy
    if capacity < total_weight:
        print("Kapacita batohu je menší než celkový objem předmětů!")

    # výpis celkového kapacity a hodnoty předmětů
    print("Celková hodnota:", total_value)
    print("Celková kapacita:", total_weight)
    print("\n")

    # návratová hodnota - hodnoty hodnot, váh a kapacity
    return values, weights, capacity

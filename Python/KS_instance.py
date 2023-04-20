import random


def generate_knapsack_instance():
    # nastavení počtu předmětů
    n_items = 20
    # nastavení kapacity batohu - 16 - 30 = 200
    capacity = 200
    # inicializace seznamů pro objemy a ceny předmětů
    volumes = []
    values = []

    # generování náhodných objemů a hodnot předmětů
    for i in range(1, n_items+1):
        volumes.append(random.randint(1, 50))
        values.append(random.randint(1, 50))

    # výpočet celkového objemu a hodnoty předmětů a výpis včetně jejich ID
    total_volume = 0
    total_value = 0
    for i in range(1, n_items+1):
        total_volume += volumes[i-1]
        total_value += values[i-1]
        print("ID:", i, "Objem:", volumes[i-1], "Cena:", values[i-1])

    # výpis celkového kapacity a hodnoty předmětů
    print("Celková hodnota:", total_value)
    print("Celková kapacita:", total_volume)
    print("\n")

    # návratová hodnota - hodnoty hodnot, váh a kapacity
    return values, volumes, capacity

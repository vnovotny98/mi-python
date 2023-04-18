import random


def generate_knapsack_instance():
    # nastavení počtu předmětů
    n_items = 20
    # nastavení kapacity batohu - 16 - 30 = 200
    capacity = 200
    # inicializace seznamů pro objemy a ceny předmětů
    volumes = []
    prices = []

    # generování náhodných objemů a cen předmětů
    for i in range(n_items):
        volumes.append(random.randint(1, 50))
        prices.append(random.randint(1, 50))

    # výpis seznamů objemů a cen předmětů
    # print("Objem:", volumes)
    # print("Cena:", prices)

    # výpočet celkového objemu a ceny předmětů a jejich ID
    total_volume = 0
    total_price = 0
    for i in range(n_items):
        total_volume += volumes[i]
        total_price += prices[i]
        print("ID:", i+1, "Objem:", volumes[i], "Cena:", prices[i])

    # kontrola, zda kapacita batohu není menší než celkový objem předmětů
    if capacity < total_volume:
        print("Kapacita batohu je menší než celkový objem předmětů!")

    # výpis celkového objemu a ceny předmětů
    print("Celkový objem:", total_volume)
    print("Celková cena:", total_price)


# volání funkce pro generování instance problému batohu
generate_knapsack_instance()

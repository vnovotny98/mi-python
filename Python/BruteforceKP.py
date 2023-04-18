import random


def generate_knapsack_instance(n_items, min_volume, max_volume, min_price, max_price, capacity):
    volumes = [random.randint(min_volume, max_volume) for _ in range(n_items)]
    prices = [random.randint(min_price, max_price) for _ in range(n_items)]
    return volumes, prices, capacity


def bruteforce_knapsack(volumes, prices, capacity):
    n_items = len(volumes)  # počet předmětů
    best_price = 0  # nejlepší nalezená cena
    best_combination = []  # nejlepší nalezená kombinace předmětů

    # iterace přes všechny možné kombinace předmětů
    for i in range(2 ** n_items):
        # inicializace objemu a ceny aktuální kombinace předmětů
        current_volume = 0
        current_price = 0
        current_combination = []

        # iterace přes všechny předměty a rozhodnutí, zda je v kombinaci nebo ne
        for j in range(n_items):
            if i & (1 << j):
                current_volume += volumes[j]
                current_price += prices[j]
                current_combination.append(j)

        # kontrola, zda aktuální kombinace předmětů splňuje kapacitu
        if current_volume <= capacity:
            # aktualizace nejlepší nalezené kombinace předmětů
            if current_price > best_price:
                best_price = current_price
                best_combination = current_combination

    # výpis nejlepší nalezené kombinace předmětů a její ceny
    print("Nejlepší kombinace předmětů:", best_combination)
    print("Cena kombinace předmětů:", best_price)


# vygenerování instance problému batohu
n_items = 20
min_volume = 1
max_volume = 50
min_price = 1
max_price = 50
capacity = 200
volumes, prices, capacity = generate_knapsack_instance(n_items, min_volume, max_volume, min_price, max_price, capacity)

# hledání řešení hrubou silou
bruteforce_knapsack(volumes, prices, capacity)

import random


# vytvoření instance problému batohu s náhodnými hodnotami
def generate_knapsack_instance(n_items, max_volume, max_price):
    volumes = [random.randint(1, max_volume) for i in range(n_items)]
    prices = [random.randint(1, max_price) for i in range(n_items)]
    capacity = max_volume * n_items // 2 # nastavení kapacity na polovinu celkového objemu
    return volumes, prices, capacity


# účelová funkce pro výpočet ceny zadané kombinace předmětů
def compute_combination_price(volumes, prices, combination):
    volume = sum([volumes[i] for i in combination])
    price = sum([prices[i] for i in combination])
    return volume, price


# hledání nejlepší kombinace předmětů hrubou silou
def bruteforce_knapsack(volumes, prices, capacity):
    n_items = len(volumes) # počet předmětů
    best_price = 0 # nejlepší nalezená cena
    best_combination = [] # nejlepší nalezená kombinace předmětů

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


entry_n_items = 20
entry_max_volume = 50
entry_max_price = 50
entry_capacity = 200

# testování
out_volumes, out_prices, out_capacity = generate_knapsack_instance(entry_n_items, entry_max_volume, entry_max_price)
bruteforce_knapsack(out_volumes, out_prices, entry_capacity)

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# define 1st Dejong function
def dejong1(x):
    return sum(xi ** 2 for xi in x)


# define 2nd Dejong function
def dejong2(x):
    assert len(x) >= 2
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


# define Schwefel function
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


# Definice funkce pro generování náhodného souseda
def generate_neighbor(x, delta=0.5):
    return [xi + random.uniform(-delta, delta) for xi in x]


# Definice funkce redukce teploty
def alfa(T, alpha_factor=0.95):
    return T * alpha_factor


# Simulované žíhání
def simulated_annealing(M, x0, N, f, T0, Tf, alpha, nT):
    # Inicializace: Nastavíme teplotu T na T0 a počáteční řešení x na x0.
    x = x0
    T = T0
    best_x = x
    fitness_progress = []
    # Opakování cyklu pro každou teplotu T, až do dosažení kritéria ukončení:
    while T > Tf:
        # 2.1. Opakování pro každou teplotu T:
        # 2.1.1. Opakování Metropolisova algoritmu nT krát pro aktuální teplotu T:
        for i in range(nT):
            # 2.1.1.1. Vygenerujeme náhodné sousední řešení x_new z aktuálního řešení x.
            x_new = generate_neighbor(x)
            # 2.1.1.2. Vypočítáme hodnotu rozdílu účelové funkce delta_f mezi aktuálním a novým řešením x a x_new.
            delta_f = f(x_new) - f(x)
            # 2.1.1.3. Pokud je delta_f menší nebo rovno nule, přijmeme nové řešení x_new jako aktuální řešení x.
            if delta_f <= 0:
                x = x_new
                if f(x) < f(best_x):
                    best_x = x
                    fitness_progress.append(best_x)
            # 2.1.1.4. Pokud je delta_f větší než nula, vygenerujeme náhodné číslo r z intervalu [0,1] a pokud r je
            # menší nebo rovno exp, přijmeme nové řešení x_new jako aktuální řešení, jinak ponecháme stávající řešení.
            else:
                r = random.uniform(0, 1)
                if r <= math.exp(-delta_f / T):
                    x = x_new
        # 2.1.2. Snížení teploty T pomocí funkce redukce teploty alfa: T = alfa(T).
        T = alfa(T)
    # Výstup: Vrátíme nejlepší nalezené řešení.
    return best_x, fitness_progress


# Testování
M = [-5, 5] * 5  # prostor řešení
x0 = [random.uniform(-5, 5) for _ in range(5)]  # počáteční řešení
N = None  # množina sousedů (v tomto případě není potřeba)
f = dejong1  # účelová funkce
T0 = 100  # počáteční teplota
Tf = 0.01  # konečná teplota
alpha_factor = 0.95  # koeficient redukce teploty
nT = 10  # počet opakování pro každou teplotu

best_x, fitness_progress = simulated_annealing(M, x0, N, f, T0, Tf, lambda T: alfa(T, alpha_factor), nT)
print(f"Nejlepší řešení: {best_x}, hodnota účelové funkce: {f(best_x)}")


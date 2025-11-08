import os
import json
import datetime
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def initialize_population(num_features, population_size):
    """Inizializza la popolazione come lista di vettori binari (0/1)."""
    return [np.random.randint(0, 2, num_features) for _ in range(population_size)]


def fitness_function(individual, X, y):
    """Calcola la fitness di un individuo in base all’accuratezza media di cross-validation."""
    if np.sum(individual) == 0:
        return 0
    X_subset = X[:, individual == 1]
    model = LogisticRegression(max_iter=500)
    score = cross_val_score(model, X_subset, y, cv=3, scoring='accuracy').mean()
    return score


def selection(population, fitnesses):
    """Selezione proporzionale alla fitness."""
    probs = fitnesses / np.sum(fitnesses)
    indices = np.random.choice(len(population), size=len(population), p=probs)
    return [population[i] for i in indices]


def crossover(parent1, parent2):
    """Crossover a punto singolo."""
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def mutation(individual, mutation_rate=0.02):
    """Applica mutazione casuale a un individuo."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def ga_feature_selection(X, y, n_generations=5, population_size=5, save_best=True):
    """Esegue l’algoritmo genetico per la selezione delle feature."""
    num_features = X.shape[1]
    population = initialize_population(num_features, population_size)

    print(f"🔄 Starting Genetic Algorithm with {population_size} individuals and {n_generations} generations...\n")

    for generation in tqdm(range(n_generations), desc="Evolving generations"):
        fitnesses = np.array([
            fitness_function(ind, X, y)
            for ind in tqdm(population, desc=f"Evaluating generation {generation+1}", leave=False)
        ])

        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        print(f"Generation {generation+1}/{n_generations} → Best Fitness: {best_fit:.4f}")

        new_population = []
        selected = selection(population, fitnesses)
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = crossover(selected[i], selected[i + 1])
                new_population.extend([mutation(child1), mutation(child2)])
        population = new_population

    # Migliore individuo
    best_individual = population[np.argmax(fitnesses)]
    selected_indices = np.where(best_individual == 1)[0]

    print(f"\n✅ Best individual selected {len(selected_indices)} features.")

    # Salva il set di feature selezionate
    if save_best:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "selected_features"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"features_{timestamp}.json")

        with open(save_path, "w") as f:
            json.dump(selected_indices.tolist(), f, indent=4)
        print(f"💾 Saved selected features to: {save_path}")

    return selected_indices


def load_selected_features(filepath):
    """Carica un set di feature salvate in formato JSON."""
    with open(filepath, "r") as f:
        indices = np.array(json.load(f))
    print(f"📂 Loaded {len(indices)} selected features from {filepath}")
    return indices

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import random
from tqdm import tqdm

def initialize_population(num_features, population_size):
    return [np.random.randint(0, 2, num_features) for _ in range(population_size)]

def fitness_function(individual, X, y):
    # Se l’individuo non seleziona feature, fitness = 0
    if np.sum(individual) == 0:
        return 0
    X_subset = X[:, individual == 1]
    model = LogisticRegression(max_iter=500)
    score = cross_val_score(model, X_subset, y, cv=3, scoring='accuracy').mean()
    return score

def selection(population, fitnesses):
    probs = fitnesses / np.sum(fitnesses)
    indices = np.random.choice(len(population), size=len(population), p=probs)
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutation(individual, mutation_rate=0.02):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def ga_feature_selection(X, y, n_generations=10, population_size=15):
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

    best_individual = population[np.argmax(fitnesses)]
    selected_indices = np.where(best_individual == 1)[0]
    print(f"\n✅ Best individual selected {len(selected_indices)} features.")
    return selected_indices

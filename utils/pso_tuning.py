import os
import json
import numpy as np
import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from models.dnn_model import create_dnn
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K


def pso_tune_dnn(X_train, y_train, X_val, y_val, n_particles=8, n_iterations=10):
    """
    Ottimizzazione degli iperparametri del modello DNN tramite PSO (Particle Swarm Optimization).
    Ottimizza:
        - learning rate
        - dropout1, dropout2, dropout3
        - batch size
    """

    # === Definizione dei limiti di ricerca per ciascun parametro ===
    param_bounds = {
        "lr": (1e-5, 1e-2),
        "dropout1": (0.1, 0.6),
        "dropout2": (0.1, 0.6),
        "dropout3": (0.1, 0.6),
        "batch_size": (16, 128),
    }

    # === Inizializzazione delle particelle ===
    particles = []
    velocities = []
    for _ in range(n_particles):
        particle = {p: np.random.uniform(low, high) for p, (low, high) in param_bounds.items()}
        velocity = {p: np.random.uniform(-abs(high - low), abs(high - low)) * 0.1 for p, (low, high) in param_bounds.items()}
        particles.append(particle)
        velocities.append(velocity)

    personal_best = particles.copy()
    personal_best_scores = [-np.inf] * n_particles

    global_best = None
    global_best_score = -np.inf

    # === Parametri PSO ===
    w = 0.6   # inerzia
    c1 = 1.5  # componente cognitiva
    c2 = 1.5  # componente sociale

    print(f"\n=== 🔧 PSO Hyperparameter Optimization Started ===")
    print(f"Particles: {n_particles}, Iterations: {n_iterations}\n")

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        for i, particle in enumerate(particles):
            # === Costruisci e valuta modello ===
            K.clear_session()

            dropout_rates = [particle["dropout1"], particle["dropout2"], particle["dropout3"]]
            lr = particle["lr"]
            batch_size = int(particle["batch_size"])

            model = create_dnn(input_dim=X_train.shape[1], num_classes=y_train.shape[1], dropout_rates=dropout_rates)
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=15,
                batch_size=batch_size,
                verbose=0,
                callbacks=[es],
            )

            val_acc = max(history.history["val_accuracy"])
            print(f"  Particle {i+1}: val_acc={val_acc:.4f}")

            # === Aggiornamento del best personale e globale ===
            if val_acc > personal_best_scores[i]:
                personal_best[i] = particle.copy()
                personal_best_scores[i] = val_acc

            if val_acc > global_best_score:
                global_best = particle.copy()
                global_best_score = val_acc

        # === Aggiornamento velocità e posizione ===
        for i, particle in enumerate(particles):
            for param, (low, high) in param_bounds.items():
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = c1 * r1 * (personal_best[i][param] - particle[param])
                social = c2 * r2 * (global_best[param] - particle[param])
                velocities[i][param] = w * velocities[i][param] + cognitive + social
                particle[param] += velocities[i][param]
                particle[param] = np.clip(particle[param], low, high)

        print(f"  → Best score so far: {global_best_score:.4f}")

    print("\n=== ✅ PSO Optimization Completed ===")
    print(f"Best validation accuracy: {global_best_score:.4f}")
    print("Best parameters found:")
    for k, v in global_best.items():
        print(f"  {k}: {v}")

    # === Salvataggio parametri migliori ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "optimized_hyperparams"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_params_{timestamp}.json")

    with open(save_path, "w") as f:
        json.dump(global_best, f, indent=4)

    print(f"\n💾 Saved best hyperparameters to {save_path}")

    return global_best


def load_pso_params(path):
    """Carica un set di iperparametri PSO da file JSON."""
    with open(path, "r") as f:
        params = json.load(f)
    return params

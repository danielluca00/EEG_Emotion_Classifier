import os
import sys
import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from utils.data_loader import load_data
from utils.plot_utils import plot_training_history, plot_confusion
from utils.feature_selection import ga_feature_selection, load_selected_features
from utils.pso_tuning import pso_tune_dnn, load_pso_params
from models.dnn_model import create_dnn
from sklearn.metrics import classification_report


def main():
    # === Create timestamped results folder ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # === Redirect stdout and stderr to a log file ===
    log_path = os.path.join(results_dir, "training_log.txt")
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout

    print(f"=== Training session started: {timestamp} ===\n")

    # === Load Data ===
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data("data/emotions.csv")

    # === Feature Selection with GA ===
    apply_ga = True
    selected_indices = None

    if apply_ga:
        features_dir = "selected_features"
        os.makedirs(features_dir, exist_ok=True)
        existing_files = sorted(
            [f for f in os.listdir(features_dir) if f.endswith(".json")],
            reverse=True
        )

        if existing_files:
            print("📂 Found saved feature sets:")
            for i, fname in enumerate(existing_files, start=1):
                print(f"  {i}. {fname}")

            choice = input("\nReuse an existing feature set? (y/n): ").strip().lower()
            if choice == "y":
                index = input("➡️  Enter set number (default = 1): ").strip()
                try:
                    index = int(index) - 1 if index else 0
                    path = os.path.join(features_dir, existing_files[index])
                    selected_indices = load_selected_features(path)
                except (ValueError, IndexError):
                    print("⚠️ Invalid choice, running new optimization...")

        if selected_indices is None:
            print("\n🧬 Running Genetic Algorithm for feature selection...")
            y_train_labels = np.argmax(y_train.values, axis=1)
            selected_indices = ga_feature_selection(X_train, y_train_labels)

        X_train = X_train[:, selected_indices]
        X_val = X_val[:, selected_indices]
        X_test = X_test[:, selected_indices]

        selected_features_path = os.path.join(results_dir, "selected_features.txt")
        with open(selected_features_path, "w") as f:
            f.write(f"Total selected features: {len(selected_indices)}\n\n")
            f.write("Selected features:\n")
            for i in selected_indices:
                f.write(f"- {i}: {feature_names[i]}\n")

        print(f"🧬 Selected features saved in: {selected_features_path}\n")

    else:
        print("\nSkipping GA feature selection (apply_ga=False).")

    # === PSO Hyperparameter Optimization ===
    use_pso = input("\n⚙️  Run PSO for hyperparameter tuning? (y/n): ").strip().lower() == "y"
    best_params = None

    params_dir = "optimized_hyperparams"
    os.makedirs(params_dir, exist_ok=True)
    existing_param_files = sorted(
        [f for f in os.listdir(params_dir) if f.endswith(".json")],
        reverse=True
    )

    if existing_param_files and not use_pso:
        print("📂 Found saved PSO hyperparameter sets:")
        for i, fname in enumerate(existing_param_files, start=1):
            print(f"  {i}. {fname}")

        choice = input("\nReuse an existing hyperparameter set? (y/n): ").strip().lower()
        if choice == "y":
            index = input("➡️  Enter set number (default = 1): ").strip()
            try:
                index = int(index) - 1 if index else 0
                path = os.path.join(params_dir, existing_param_files[index])
                best_params = load_pso_params(path)
            except (ValueError, IndexError):
                print("⚠️ Invalid choice, skipping to default parameters.")
                best_params = None

    if use_pso:
        best_params = pso_tune_dnn(X_train, y_train, X_val, y_val)

    # === Set Hyperparameters ===
    if best_params:
        lr = best_params["lr"]
        batch_size = int(best_params["batch_size"])
        dropout_rates = [best_params["dropout1"], best_params["dropout2"], best_params["dropout3"]]
    else:
        lr = 0.001
        batch_size = 32
        dropout_rates = [0.3, 0.3, 0.25]

    # === Create and Train Model ===
    print("Creating DNN model...")
    model = create_dnn(input_dim=X_train.shape[1], num_classes=y_train.shape[1], dropout_rates=dropout_rates)
    model.summary()

    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_path = os.path.join(results_dir, "best_dnn_model.h5")
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    lr_schedule = LearningRateScheduler(lambda epoch: lr * np.exp(-epoch / 10.))

    print("\n🚀 Starting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[es, mc, lr_schedule],
        verbose=1
    )

    plot_training_history(history, save_path=os.path.join(results_dir, "training_history.png"))

    # === Evaluate ===
    print("\nEvaluating model...")
    best_model = load_model(model_path)
    acc = best_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    y_true = y_test.idxmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

    print("\nClassification Report:\n")
    print(report)

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    plot_confusion(
        y_true,
        y_pred,
        classes=['Negative', 'Neutral', 'Positive'],
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )

    print(f"\n✅ All results saved in: {results_dir}")
    print(f"=== Training session ended at {datetime.datetime.now().strftime('%H:%M:%S')} ===")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"\n📝 Full terminal output saved in: {log_path}")


if __name__ == "__main__":
    main()

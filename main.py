import os
import sys
import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from utils.data_loader import load_data
from utils.plot_utils import plot_training_history, plot_confusion
from utils.feature_selection import ga_feature_selection
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
    sys.stderr = sys.stdout  # redirect errors as well

    print(f"=== Training session started: {timestamp} ===\n")

    # === Load Data ===
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data("data/emotions.csv")

    # === Optional: Feature Selection with GA ===
    apply_ga = True  # set to False to skip GA phase

    if apply_ga:
        print("\nRunning Genetic Algorithm for feature selection...")
        y_train_labels = np.argmax(y_train.values, axis=1)

        selected_indices = ga_feature_selection(
            X_train,
            y_train_labels,
            n_generations=10,
            population_size=15
        )

        # Apply feature selection
        X_train = X_train[:, selected_indices]
        X_val = X_val[:, selected_indices]
        X_test = X_test[:, selected_indices]

        print(f"Reduced feature set shape: {X_train.shape}")

        # Save selected features
        selected_features_path = os.path.join(results_dir, "selected_features.txt")
        with open(selected_features_path, "w") as f:
            f.write(f"Total selected features: {len(selected_indices)}\n\n")
            f.write("Selected features:\n")
            for i in selected_indices:
                f.write(f"- {i}: {feature_names[i]}\n")

        print(f"🧬 Selected features saved in: {selected_features_path}\n")

    # === Create Model ===
    print("Creating DNN model...")
    model = create_dnn(input_dim=X_train.shape[1], num_classes=y_train.shape[1])
    model.summary()

    # === Compile ===
    print("\nCompiling model...")
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # === Callbacks ===
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_path = os.path.join(results_dir, "best_dnn_model.h5")
    mc = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # === Train ===
    print("\nStarting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es, mc, lr_schedule],
        verbose=1
    )

    # === Save training plots ===
    print("\nSaving training plots...")
    plot_training_history(history, save_path=os.path.join(results_dir, "training_history.png"))

    # === Evaluate ===
    print("\nEvaluating best model on test set...")
    best_model = load_model(model_path)
    acc = best_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    y_true = y_test.idxmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

    print("\nClassification Report:\n")
    print(report)

    # === Save classification report ===
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # === Save confusion matrix plot ===
    print("\nSaving confusion matrix...")
    plot_confusion(
        y_true,
        y_pred,
        classes=['Negative', 'Neutral', 'Positive'],
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )

    print(f"\n✅ All results saved in: {results_dir}")
    print(f"=== Training session ended at {datetime.datetime.now().strftime('%H:%M:%S')} ===")

    # === Close log and restore stdout ===
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"\n📝 Full terminal output saved in: {log_path}")


if __name__ == "__main__":
    main()

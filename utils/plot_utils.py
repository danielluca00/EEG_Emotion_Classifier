import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import os


def save_plot(fig, path):
    """Helper function to save matplotlib figures."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_training_history(history, save_path=None):
    """Plot training accuracy and loss; save if path provided."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0].plot(acc, 'r', label='Train Acc')
    ax[0].plot(val_acc, 'b', label='Val Acc')
    ax[0].legend()
    ax[0].set_title("Accuracy")

    ax[1].plot(loss, 'r', label='Train Loss')
    ax[1].plot(val_loss, 'b', label='Val Loss')
    ax[1].legend()
    ax[1].set_title("Loss")

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()


def plot_confusion(y_true, y_pred, classes, save_path=None):
    """Plot and optionally save normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()

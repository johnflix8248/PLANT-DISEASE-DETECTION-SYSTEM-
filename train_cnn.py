"""
Train a CNN for plant disease detection (Spyder: run this file).

Saves:
  models/plant_disease_cnn.keras
  models/class_names.json
  models/training_history.json
  models/accuracy_graph.png
  models/loss_graph.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from download_dataset import DATA_ROOT, download_dataset

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12
MODEL_DIR = Path(__file__).resolve().parent / "models"
SEED = 42


def load_datasets():
    download_dataset()
    kwargs = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(DATA_ROOT / "train", **kwargs)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT / "validation", shuffle=False, **kwargs
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=IMG_SIZE + (3,)),
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_graphs(history) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], marker="o", label="Train accuracy")
    plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "accuracy_graph.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], marker="o", label="Train loss")
    plt.plot(history.history["val_loss"], marker="o", label="Validation loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "loss_graph.png", dpi=140)
    plt.close()


def main():
    train_ds, val_ds, class_names = load_datasets()
    print("Classes:", class_names)
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "plant_disease_cnn.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    save_graphs(history)
    (MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (MODEL_DIR / "training_history.json").write_text(json.dumps(hist, indent=2))
    model.save(MODEL_DIR / "plant_disease_cnn.keras")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {val_acc * 100:.2f}%")
    print(f"Saved model and graphs in {MODEL_DIR}")


if __name__ == "__main__":
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    main()

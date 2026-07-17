from __future__ import annotations

from typing import Literal

ModelName = Literal["cnn", "lstm", "gru"]


def build_model(
    name: ModelName,
    input_shape: tuple[int, int] = (128, 9),
    num_classes: int = 6,
    learning_rate: float = 1e-3,
):
    """Build a compact deep-learning baseline for UCI HAR.

    TensorFlow is imported lazily so data utilities remain usable without it.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for model training. Install with `pip install -e .[train]`."
        ) from exc

    if name == "cnn":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name="uci_har_cnn",
        )
    elif name in {"lstm", "gru"}:
        recurrent = tf.keras.layers.LSTM if name == "lstm" else tf.keras.layers.GRU
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                recurrent(64),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name=f"uci_har_{name}",
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose cnn, lstm, or gru.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

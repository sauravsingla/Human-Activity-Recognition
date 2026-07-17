from __future__ import annotations

from typing import Literal

ModelName = Literal["cnn", "lstm", "gru"]
ModelConfiguration = Literal["paper", "modern"]


def _tensorflow():
    """Import TensorFlow lazily so data utilities work without it."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for model training. "
            "Install with `pip install -e \".[train]\"`."
        ) from exc
    return tf


def build_paper_model(
    name: ModelName,
    input_shape: tuple[int, int] = (128, 9),
    num_classes: int = 6,
    learning_rate: float = 1e-3,
):
    """Build the architectures represented by the original paper notebooks.

    The paper configuration intentionally preserves the compact 32-unit
    recurrent networks and the two-layer 1D CNN used in the historical
    experiments. Labels are zero-based, so sparse categorical cross-entropy is
    equivalent to the notebooks' one-hot categorical objective.
    """
    tf = _tensorflow()

    if name == "cnn":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=3,
                    activation="relu",
                    kernel_initializer="he_uniform",
                ),
                tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=3,
                    activation="relu",
                    kernel_initializer="he_uniform",
                ),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name="paper_uci_har_cnn",
        )
    elif name in {"lstm", "gru"}:
        recurrent = tf.keras.layers.LSTM if name == "lstm" else tf.keras.layers.GRU
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                recurrent(32),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ],
            name=f"paper_uci_har_{name}",
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose cnn, lstm, or gru.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_modern_model(
    name: ModelName,
    input_shape: tuple[int, int] = (128, 9),
    num_classes: int = 6,
    learning_rate: float = 1e-3,
):
    """Build stronger contemporary baselines without claiming paper fidelity."""
    tf = _tensorflow()

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
            name="modern_uci_har_cnn",
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
            name=f"modern_uci_har_{name}",
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose cnn, lstm, or gru.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(
    name: ModelName,
    input_shape: tuple[int, int] = (128, 9),
    num_classes: int = 6,
    learning_rate: float = 1e-3,
    configuration: ModelConfiguration = "paper",
):
    """Build a paper-reproduction or modern HAR model."""
    if configuration == "paper":
        return build_paper_model(name, input_shape, num_classes, learning_rate)
    if configuration == "modern":
        return build_modern_model(name, input_shape, num_classes, learning_rate)
    raise ValueError("configuration must be either 'paper' or 'modern'")

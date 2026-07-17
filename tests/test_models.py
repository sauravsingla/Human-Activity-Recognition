import pytest


tf = pytest.importorskip("tensorflow")

from har.models import build_model  # noqa: E402


@pytest.mark.parametrize("name", ["cnn", "lstm", "gru"])
@pytest.mark.parametrize("configuration", ["paper", "modern"])
def test_model_output_shape(name: str, configuration: str) -> None:
    model = build_model(name, configuration=configuration)
    assert model.output_shape == (None, 6)
    assert model.input_shape == (None, 128, 9)
    assert model.count_params() > 0


def test_paper_recurrent_models_use_32_units() -> None:
    for name, layer_type in (("lstm", tf.keras.layers.LSTM), ("gru", tf.keras.layers.GRU)):
        model = build_model(name, configuration="paper")
        recurrent_layers = [layer for layer in model.layers if isinstance(layer, layer_type)]
        assert len(recurrent_layers) == 1
        assert recurrent_layers[0].units == 32


def test_paper_cnn_matches_historical_filter_layout() -> None:
    model = build_model("cnn", configuration="paper")
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
    assert [layer.filters for layer in conv_layers] == [32, 32]
    assert [layer.kernel_size for layer in conv_layers] == [(3,), (3,)]


def test_invalid_model_and_configuration_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="configuration"):
        build_model("cnn", configuration="invalid")  # type: ignore[arg-type]

from tensorflow import keras


# Registra a camada para permitir o carregamento dos modelos salvos.
@keras.utils.register_keras_serializable(package="RUL")
class PositionalEmbedding(keras.layers.Layer):
    """Embedding temporal treinável somado à projeção das features."""

    def __init__(self, window_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.d_model = int(d_model)

    # Cria um vetor treinável para cada posição temporal.
    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.window_size, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    # Adiciona a informação de posição à entrada.
    def call(self, inputs):
        return inputs + self.position_embeddings

    # Guarda os parâmetros necessários para reconstruir a camada.
    def get_config(self):
        return {
            **super().get_config(),
            "window_size": self.window_size,
            "d_model": self.d_model,
        }

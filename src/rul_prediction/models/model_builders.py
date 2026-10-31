from tensorflow import keras
from src.rul_prediction.models.transformer_layers import PositionalEmbedding


# Constrói e compila a CNN-LSTM com os parâmetros recebidos.
def build_cnn_lstm(input_shape, params):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(filters=params['filters'],
                            kernel_size=params['kernel_size'], activation='relu'),
        keras.layers.LSTM(units=params['lstm_units'], return_sequences=False),
        keras.layers.Dropout(params['dropout_rate']),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse', metrics=['mae'],
    )
    return model


def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout):
    """Mantém d_model nas duas conexões residuais."""
    if d_model % num_heads != 0:
        raise ValueError('d_model deve ser divisível por num_heads.')
    # A atenção relaciona os diferentes ciclos da janela.
    attention = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        output_shape=d_model, dropout=dropout,
    )(inputs, inputs)
    attention = keras.layers.Dropout(dropout)(attention)
    x = keras.layers.Add()([inputs, attention])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Transforma a representação e retorna à dimensão do residual.
    ff = keras.layers.Dense(ff_dim, activation='relu')(x)
    ff = keras.layers.Dropout(dropout)(ff)
    ff = keras.layers.Dense(d_model)(ff)
    ff = keras.layers.Dropout(dropout)(ff)
    x = keras.layers.Add()([x, ff])
    return keras.layers.LayerNormalization(epsilon=1e-6)(x)


# Constrói e compila o Transformer sem iniciar o treino.
def build_transformer(input_shape, params):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(params['d_model'])(inputs)
    x = PositionalEmbedding(input_shape[0], params['d_model'])(x)
    for _ in range(params['num_layers']):
        x = transformer_encoder(
            x, params['d_model'], params['num_heads'],
            params['ff_dim'], params['dropout_rate'],
        )
    # Resume os ciclos da janela em um vetor para prever o RUL.
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(params['dropout_rate'])(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name='transformer_rul')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse', metrics=['mae'],
    )
    return model

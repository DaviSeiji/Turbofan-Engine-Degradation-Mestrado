"""Builder do LSTM Autoencoder: entrada e saída com o mesmo formato."""

from tensorflow import keras


def build_lstm_autoencoder(input_shape, params):
    layers = int(params['num_layers'])
    if layers not in (1, 2):
        raise ValueError('num_layers deve ser 1 ou 2.')
    units = params['encoder_units']
    latent = params['latent_dim']
    dropout = params['dropout_rate']
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    if layers == 2:
        x = keras.layers.LSTM(units, return_sequences=True)(x)
        x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LSTM(units, return_sequences=False)(x)
    # Bottleneck linear; o decoder recebe apenas essa representação.
    x = keras.layers.Dense(latent, name='latent_representation')(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.RepeatVector(input_shape[0])(x)
    for _ in range(layers):
        x = keras.layers.LSTM(units, return_sequences=True)(x)
        x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(input_shape[1]))(x)
    model = keras.Model(inputs, outputs, name='lstm_autoencoder')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

#!/usr/bin/env python3
"""
Defines a function that creates a vanilla autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Args:
        input_dims: integer containing dimensions of model input
        hidden_layers: list containing number of nodes for each hidden layer
        latent_dims: integer containing dimensions of latent space

    Returns:
        encoder, decoder, auto
    """
    if not isinstance(input_dims, int):
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if not isinstance(hidden_layers, list):
        raise TypeError("hidden_layers must be a list of ints "
                        "representing number of nodes for each layer")
    for nodes in hidden_layers:
        if not isinstance(nodes, int):
            raise TypeError("hidden_layers must be a list of ints "
                            "representing number of nodes for each layer")
    if not isinstance(latent_dims, int):
        raise TypeError("latent_dims must be an int containing dimensions "
                        "of latent space representation")

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=inputs, outputs=latent)

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Full Autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='mean_squared_error')

    return encoder, decoder, auto

#!/usr/bin/env python3
"""
Defines function that creates a vanilla autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder
    """
    if type(input_dims) is not int:
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if type(hidden_layers) is not list:
        raise TypeError("hidden_layers must be a list of ints \
            representing number of nodes for each layer")
    for nodes in hidden_layers:
        if type(nodes) is not int:
            raise TypeError("hidden_layers must be a list of ints \
                representing number of nodes for each layer")
    if type(latent_dims) is not int:
        raise TypeError("latent_dims must be an int containing dimensions \
            of latent space representation")

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoder_value = encoder_inputs

    for i in range(len(hidden_layers)):
        encoder_layer = keras.layers.Dense(hidden_layers[i],
                                           activation='relu')
        encoder_value = encoder_layer(encoder_value)

    latent_layer = keras.layers.Dense(latent_dims, activation='relu')
    latent_outputs = latent_layer(encoder_value)

    encoder = keras.Model(inputs=encoder_inputs, outputs=latent_outputs)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoder_value = decoder_inputs

    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = keras.layers.Dense(units=hidden_layers[i],
                                           activation='relu')
        decoder_value = decoder_layer(decoder_value)

    decoder_output_layer = keras.layers.Dense(units=input_dims,
                                              activation='sigmoid')
    decoder_outputs = decoder_output_layer(decoder_value)

    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # Autoencoder
    auto_inputs = encoder_inputs
    encoded_representation = encoder(auto_inputs)
    reconstructed_outputs = decoder(encoded_representation)

    auto = keras.Model(inputs=auto_inputs, outputs=reconstructed_outputs)
    auto.compile(optimizer='adam', loss='mean_squared_error')

    return encoder, decoder, auto

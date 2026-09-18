#!/usr/bin/env python3
"""
Defines function that creates a vanilla autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoder_value = encoder_inputs

    for nodes in hidden_layers:
        encoder_value = keras.layers.Dense(nodes, activation='relu')(encoder_value)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoder_value)
    encoder = keras.Model(inputs=encoder_inputs, outputs=latent)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoder_value = decoder_inputs

    for nodes in reversed(hidden_layers):
        decoder_value = keras.layers.Dense(nodes, activation='relu')(decoder_value)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoder_value)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # Combined Autoencoder
    auto_inputs = encoder_inputs
    auto_outputs = decoder(encoder(auto_inputs))
    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

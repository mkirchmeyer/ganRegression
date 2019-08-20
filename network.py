import keras
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, concatenate


def build_generator(network):
    seed = network.seed
    random_normal = keras.initializers.RandomNormal(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    # linear and sinus datasets
    if network.architecture == 1:
        # This will input x & noise and will output Y.
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(noise_output)

        concat = concatenate([x_output, noise_output])
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x_output)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)
        noise_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(noise_output)

        concat = concatenate([x_output, noise_output])
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(x)

        noise = Input(shape=(network.z_input_size,))
        noise_output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(noise)

        concat = concatenate([x_output, noise_output])

        output = Dense(150, activation=activation, kernel_initializer=kerner_initializer)(concat)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
        output = Dense(network.y_input_size, activation="linear", kernel_initializer=random_normal)(output)

        model = Model(inputs=[noise, x], outputs=output)

    else:
        raise NotImplementedError("Architecture does not exist")

    return model


def build_discriminator(network):
    seed = network.seed
    random_uniform = keras.initializers.RandomUniform(seed=seed)

    if network.activation == "linear":
        activation = "linear"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "elu":
        activation = "elu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "selu":
        activation = "selu"
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "relu":
        activation = "relu"
        kerner_initializer = keras.initializers.he_uniform(seed=seed)
    elif network.activation == "lrelu":
        activation = LeakyReLU()
        kerner_initializer = keras.initializers.he_normal(seed=seed)
    elif network.activation == "tanh":
        activation = "tanh"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    elif network.activation == "sigmoid":
        activation = "sigmoid"
        kerner_initializer = keras.initializers.RandomUniform(seed=seed)
    else:
        raise NotImplementedError("Activation not recognized")

    # linear and sinus datasets
    if network.architecture == 1:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(15, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    # heteroscedastic, exp and multi-modal datasets
    elif network.architecture == 2:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(40, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    # CA-housing and ailerons
    elif network.architecture == 3:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 4:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(25, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    elif network.architecture == 5:
        x = Input(shape=(network.x_input_size,), dtype='float')
        x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

        label = Input(shape=(network.y_input_size,))
        label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

        concat = concatenate([x_output, label_output])
        concat = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        concat = Dense(50, activation=activation, kernel_initializer=kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)

        model = Model(inputs=[x, label], outputs=validity)

    else:
        raise NotImplementedError("Architecture does not exist")

    return model

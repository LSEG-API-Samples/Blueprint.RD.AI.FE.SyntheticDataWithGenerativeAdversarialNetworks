import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def noise(dimensions, n):
    return randn(dimensions*n).reshape(n, dimensions)


def get_real_sample(n, df):
    x = randn(n)
    y = x * x
    x = x.reshape(n, 1)
    y = y.reshape(n, 1)
    X = np.hstack((x, y))
    c = np.ones(n).reshape(n, 1)
    return X, c


def get_actual_sample(n):
    x = randn(n)
    y = x * x
    x = x.reshape(n, 1)
    y = y.reshape(n, 1)
    X = np.hstack((x, y))
    c = np.ones(n).reshape(n, 1)
    return X, c


def get_synthetic_sample(generator: Sequential, noise_dimensions, n):
    x = noise(noise_dimensions, n)
    X = generator.predict(x)
    c = np.zeros(n).reshape(n, 1)

    return X, c


def discriminator_model(neurons=50, input_dimensions=2):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=input_dimensions))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def generator_model(neurons=50, input_dimensions=2, output_dimensions=2):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=input_dimensions))
    model.add(Dense(output_dimensions, activation='linear'))

    return model


def GAN(discriminator, generator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def epoch_results(epoch, generator: Sequential, discriminator: Sequential, noise_dimensions, features_dictionary, n=200):
    _, actual_acc = discriminator.evaluate(features_dictionary['x'], features_dictionary['c'])
    x_synthetic, y_synthetic = get_synthetic_sample(generator, noise_dimensions, n)
    _, synthetic_acc = discriminator.evaluate(x_synthetic, y_synthetic)

    print(f'Epoch:{epoch}, actual accuracy:{actual_acc}, synthetic accuracy:{synthetic_acc}.')

    plt.figure()
    plt.scatter(features_dictionary['x'][:, 0], features_dictionary['x'][:, 1], color='blue')
    plt.scatter(x_synthetic[:, 0], x_synthetic[:, 1], color='red')
    plt.show()

    return x_synthetic, y_synthetic


def train(generator: Sequential, discriminator: Sequential, gan: Sequential, noise_dimensions, features_dictionary,
          epochs=100000, batch_size=512):
    for i in range(epochs):
        x_synthetic, y_synthetic = get_synthetic_sample(generator, noise_dimensions, batch_size)
        discriminator.train_on_batch(features_dictionary['x'], features_dictionary['c'])
        discriminator.train_on_batch(x_synthetic, y_synthetic)

        x_gan = noise(noise_dimensions, batch_size)
        y_gan = np.ones((batch_size, 1))

        gan.train_on_batch(x_gan, y_gan)
        if i % 5000 == 0:
            x_synthetic, y_synthetic = epoch_results(i, generator, discriminator, noise_dimensions,
                                                 features_dictionary, batch_size)

    return x_synthetic, y_synthetic

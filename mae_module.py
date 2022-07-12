from gan import generator_model, discriminator_model, GAN, train
from utilities import load_pickled_object

features_file_path = './studies/flash_crashes/omx30/features/'


def mae_phase():
    features_dictionary = load_pickled_object(features_file_path + 'x_c.pickle')

    noise_dimensions = 2
    generator = generator_model(noise_dimensions)
    discriminator = discriminator_model()
    gan = GAN(discriminator, generator)
    train(generator, discriminator, gan, noise_dimensions, features_dictionary)

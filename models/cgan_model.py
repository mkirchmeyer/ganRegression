from __future__ import print_function, division

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from network import build_discriminator, build_generator

"""
CGAN code derived from https://github.com/eriklindernoren/Keras-GAN
Generator will input (x & noise) and will output Ypred.
Discriminator will input (x & Ypred) and will differentiate between that and Y.
"""


class CGAN():
    def __init__(self, exp_config):
        if exp_config.model.optim_gen == "Adam":
            self.optimizer_gen = Adam(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        else:
            self.optimizer_gen = SGD(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        if exp_config.model.optim_disc == "Adam":
            self.optimizer_disc = Adam(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        else:
            self.optimizer_disc = SGD(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        self.activation = exp_config.model.activation
        self.seed = exp_config.model.random_seed
        self.scenario = exp_config.dataset.scenario

        if self.scenario == "CA-housing":
            self.x_input_size = 8
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        elif self.scenario == "ailerons":
            self.x_input_size = 40
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        else:
            self.x_input_size = 1
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            if self.scenario == "linear" or self.scenario == "sinus":
                self.architecture = 1
            else:
                self.architecture = 2

        if exp_config.model.architecture is not None:
            self.architecture = exp_config.model.architecture

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self)
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_disc,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator(self)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.z_input_size,))

        x = Input(shape=(self.x_input_size,))
        label = self.generator([noise, x])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        validity = self.discriminator([x, label])

        # The combined model (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, x], validity)
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_gen)

    def train(self, xtrain, ytrain, epochs, batch_size=128, verbose=True):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        dLossErr = np.zeros([epochs, 1])
        dLossReal = np.zeros([epochs, 1])
        dLossFake = np.zeros([epochs, 1])
        gLossErr = np.zeros([epochs, 1])
        genPred = np.zeros([epochs, 1])
        genReal = np.zeros([epochs, 1])

        for epoch in range(epochs):
            for batch_idx in range(int(xtrain.shape[0] // batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random half batch of images
                idx = np.random.randint(0, xtrain.shape[0], batch_size)
                x, true_labels = xtrain[idx], ytrain[idx]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.z_input_size))
                # Generate a half batch of new images
                fake_labels = self.generator.predict([noise, x])
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([x, true_labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([x, fake_labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                # Condition on x
                idx = np.random.randint(0, xtrain.shape[0], batch_size)
                sample = xtrain[idx]
                # Train the generator
                g_loss = self.combined.train_on_batch([noise, sample], valid)

            dLossErr[epoch] = d_loss[0]
            dLossReal[epoch] = d_loss_real[0]
            dLossFake[epoch] = d_loss_fake[0]
            gLossErr[epoch] = g_loss

            if verbose:
                print(f"Epoch: {epoch} / dLoss: {d_loss[0]} / gLoss: {g_loss}")

            ypred = self.predict(xtrain)
            genPred[epoch] = np.average(ypred)
            genReal[epoch] = np.average(ytrain)

        return dLossErr, dLossReal, dLossFake, gLossErr, genPred, genReal

    def predict(self, xtest):
        noise = np.random.normal(0, 1, (xtest.shape[0], self.z_input_size))
        ypred = self.generator.predict([noise, xtest])
        return ypred

    def sample(self, xtest, n_samples):
        y_samples_gan = self.predict(xtest)
        for i in range(n_samples - 1):
            ypred_gan = self.predict(xtest)
            y_samples_gan = np.hstack([y_samples_gan, ypred_gan])
        median = []
        mean = []
        for j in range(y_samples_gan.shape[0]):
            median.append(np.median(y_samples_gan[j, :]))
            mean.append(np.mean(y_samples_gan[j, :]))

        return np.array(mean).reshape(-1, 1), np.array(median).reshape(-1, 1), y_samples_gan

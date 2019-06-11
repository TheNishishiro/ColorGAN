from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os, sys
from PIL import Image
from glob import glob


class DCGAN():
    def __init__(self):
        self.img_rows = 384 #384
        self.img_cols = 256
        self.channels_mono = 1
        self.channels_color = 3
        self.img_shape_color = (self.img_rows, self.img_cols, self.channels_color)
        self.img_shape_mono = (self.img_rows, self.img_cols, self.channels_mono)

        self.n_batches = 0
        self.DatasetFolder = "hentai"

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.Build_Discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # Build the generator
        self.generator = self.Build_Generator()

        img_A = Input(shape=self.img_shape_color)
        img_B = Input(shape=self.img_shape_mono)

        fake_A = self.generator(img_B)

        self.discriminator.trainable = False

        valid = self.discriminator([fake_A, img_B])

        self.combined = Model([img_A, img_B], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def Build_Generator(self):
        inp = Input(shape=self.img_shape_mono)
        d0 = Conv2D(64, kernel_size=3, strides=1, padding='same')(inp)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Conv2D(128, kernel_size=3, strides=1, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Conv2D(256, kernel_size=3, strides=1, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Conv2D(512, kernel_size=3, strides=1, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        ouput = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(d0)

        return Model(inp, ouput)


    def Build_Discriminator(self):
        img_A = Input(shape=self.img_shape_color)
        img_B = Input(shape=self.img_shape_mono)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d0 = Conv2D(8, kernel_size=5, strides=2, padding='same')(combined_imgs)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Dropout(0.25)(d0)
        d0 = Conv2D(16, kernel_size=5, strides=2, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Dropout(0.25)(d0)
        d0 = Conv2D(32, kernel_size=5, strides=2, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Dropout(0.25)(d0)
        d0 = Conv2D(64, kernel_size=5, strides=2, padding='same')(d0)
        d0 = BatchNormalization(momentum=0.8)(d0)
        d0 = Activation("relu")(d0)
        d0 = Dropout(0.25)(d0)
        d0 = Flatten()(d0)
        output = Dense(1, activation="sigmoid")(d0)

        return Model([img_A, img_B], output)


    def Train(self, epochs, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.load_batch(batch_size)):

                fake_A = self.generator.predict(imgs_B)

                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], valid)

                elapsed_time = datetime.datetime.now() - start_time

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss,
                                                                        elapsed_time))

                if batch_i  % sample_interval == 0:
                    self.sample_images(str(epoch) + "_" + str(batch_i))


    def sample_images(self, epoch):
        path = glob(f'./data/{self.DatasetFolder}/*')
        batch_images = np.random.choice(path, size=3)

        conc = []
        conFinal = []
        for inp in batch_images:

            fake_A = np.expand_dims(self.loadImage(inp)[0], axis=3)
            fake_A = self.generator.predict(fake_A)
            fake_A = (0.5 * fake_A + 0.5)*255

            gray = self.loadImage(inp)[0]
            gray = np.stack((gray,)*3, axis=-1)
            gray = (0.5 * gray + 0.5)*255

            col = self.loadImage(inp)[1]
            col = (0.5 * col + 0.5)*255

            conc.append(np.vstack((gray[0], fake_A[0], col[0])))

            if(len(conc) == 2 and conFinal == []):
                conFinal = np.hstack((conc[0], conc[1]))
                conc = []
            elif(len(conc) == 1 and conFinal != []):
                conFinal = np.hstack((conc[0], conFinal))
                conc = []

        image = Image.fromarray(conFinal.astype('uint8'), 'RGB')
        image.save("./images/" + str(epoch) + ".png")

    def load_batch(self, batch_size=1):
        size = (self.img_cols, self.img_rows)
        path = glob(f'./data/{self.DatasetFolder}/*')
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                im_gray = Image.open(img).convert('L')
                im_gray = im_gray.resize(size, Image.ANTIALIAS)
                im = Image.open(img).convert('RGB')
                im = im.resize(size, Image.ANTIALIAS)
                if np.random.random() > 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    im_gray = im_gray.transpose(Image.FLIP_LEFT_RIGHT)
                
                im_gray = np.array(im_gray)
                im_gray = np.expand_dims(im_gray, axis=3)
                #im_gray = np.stack((im_gray,)*3, axis=-1)
                im = np.array(im)
                

                imgs_A.append(im)
                imgs_B.append(im_gray)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def loadImage(self, imageName):
        size = (self.img_cols, self.img_rows)
        img = []
        img2 = []

        im_gray = Image.open(imageName).convert('L')
        im_gray = im_gray.resize(size, Image.ANTIALIAS)
        im_gray = np.array(im_gray)
        #im_gray = np.expand_dims(im_gray, axis=3)
        #im_gray = np.stack((im_gray,)*3, axis=-1)
        im_gray = (im_gray / 127.5) - 1
        
        img.append(im_gray)
        img = np.array(img)

        im = Image.open(imageName).convert('RGB')
        im = im.resize(size, Image.ANTIALIAS)
        im = np.array(im)
        im = (im / 127.5) - 1
        img2.append(im)
        img2 = np.array(img2)

        return [img, img2]

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.Train(epochs=200, batch_size=1, sample_interval=20)
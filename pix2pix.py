from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
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

class Pix2Pix():
    def __init__(self, model_name, dataset_folder):
        # Input shape
        self.DatasetFolder = dataset_folder
        self.img_rows = 256 #384
        self.img_cols = 256
        self.channels_color = 3
        self.channels_mono = 3
        self.img_shape_color = (self.img_rows, self.img_cols, self.channels_color)
        self.img_shape_mono = (self.img_rows, self.img_cols, self.channels_mono)

        # Configure data loader



        # Calculate output shape of D (PatchGAN)
        patch_r = int(self.img_rows / 2**4)
        patch_c = int(self.img_cols / 2**4)
        self.disc_patch = (patch_r, patch_c, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        if(model_name == "null"):
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            # Build the generator
            self.generator = self.build_generator()
        else:
            json_file = open(f"./models/d_{model_name}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.discriminator = model_from_json(loaded_model_json)
            self.discriminator.load_weights(f"./models/d_{model_name}.h5")
            print("Loaded discriminator!")
            json_file = open(f"./models/g_{model_name}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.generator = model_from_json(loaded_model_json)
            self.generator.load_weights(f"./models/g_{model_name}.h5")
            print("Loaded generator!")

        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape_color)
        img_B = Input(shape=self.img_shape_mono)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape_mono)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels_color, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = Dropout(0.25)(d)
            return d

        img_A = Input(shape=self.img_shape_color)
        img_B = Input(shape=self.img_shape_mono)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50, model_name="hentaiGAN"):
        file= open("D_G_losses.csv","a+")
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            imgs_A, imgs_B = self.load_images(batch_size)
            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))
            file.write(f"{epoch}:{d_loss[0]}:{g_loss[0]}:{elapsed_time}\n")
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.export_model(model_name, epoch)
        file.close()

    def noise(self,image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy


    def sample_images(self, epoch):
        os.makedirs('./images/', exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.load_images(batch_size=3)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


    def load_images(self, batch_size):
        size = (self.img_cols, self.img_rows)
        i = 1

        imgs_A = []
        imgs_B = []

        path = glob(f'./data/{self.DatasetFolder}/*')
        batch_images = np.random.choice(path, size=batch_size)
        for image in batch_images:
            try:
                im_gray = Image.open(image).convert('L')
                im_gray = im_gray.resize(size, Image.ANTIALIAS)
                im = Image.open(image).convert('RGB')
                im = im.resize(size, Image.ANTIALIAS)
                if np.random.random() > 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    im_gray = im_gray.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.random() > 0.2:
                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
                    im_gray = im_gray.transpose(Image.FLIP_TOP_BOTTOM)
                
                im_gray = np.array(im_gray)
                im_gray = np.stack((im_gray,)*3, axis=-1)
                im = np.array(im)
                if np.random.random() > 0.35:
                    im = self.noise(im)
                    
                im_gray = (im_gray / 127.5) - 1
                im = (im / 127.5) - 1
                

                imgs_A.append(im)
                imgs_B.append(im_gray)
            except IOError:
                print("cannot create thumbnail for '%s'" % image)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        return imgs_A, imgs_B

    def loadInputImage(self, imageName):
        size = (self.img_cols, self.img_rows)
        img = []
        im_gray = Image.open("./input/" + imageName).convert('L')
        im_gray = im_gray.resize(size, Image.ANTIALIAS)
        im_gray = np.array(im_gray)
        im_gray = np.stack((im_gray,)*3, axis=-1)
        im_gray = (im_gray / 127.5) - 1
        img.append(im_gray)
        img = np.array(img)
        return img


            
    def export_model(self, model_name, epoch):
        # serialize model to JSON
        os.makedirs('./models/', exist_ok=True)
        model_json = self.discriminator.to_json()
        model_name = model_name + "_v" + str(int(epoch/1000))
        with open(f"./models/d_{model_name}.json", "w") as json_file:
            json_file.write(model_json)
        self.discriminator.save_weights(f"./models/d_{model_name}.h5")

        model_json = self.generator.to_json()
        with open(f"./models/g_{model_name}.json", "w") as json_file:
            json_file.write(model_json)
        self.generator.save_weights(f"./models/g_{model_name}.h5")

        print("Saved model.")

    def predict(self, imageName):
        fake_A = self.generator.predict(self.loadInputImage(imageName))
        fake_A = (0.5 * fake_A + 0.5)*255
        image = Image.fromarray(fake_A[0].astype('uint8'), 'RGB')
        image.save("./output/" + imageName)
        



if __name__ == '__main__':
    option = input('1) Train new model\n2) Predict\n3) Train existing\n>')  
    model_name = input('Enter model name: ') 
    dataset_folder = input('Enter dataset folder: ')
    if(option == "1"):
        gan = Pix2Pix("null", dataset_folder)
        gan.train(epochs=400000, batch_size=1, sample_interval=200, model_name=model_name)
    elif(option == "2"):
        gan = Pix2Pix(model_name, dataset_folder)
        picture = input('Pic name: ') 
        gan.predict(picture)
    elif(option == "3"):
        gan = Pix2Pix(model_name, dataset_folder)
        gan.train(epochs=400000, batch_size=1, sample_interval=200, model_name=model_name.split('_')[0])

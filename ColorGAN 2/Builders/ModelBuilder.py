from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras import backend as K

def BuildDiscriminator(colorImageShape, monoImageShape, filters):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = SpectralNormalization(Conv2D(filters, kernel_size=f_size, strides=2, padding='same'))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.5)(d)
        return d

    img_A = Input(shape=colorImageShape)
    img_B = Input(shape=monoImageShape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, filters, bn=False)
    d2 = d_layer(d1, filters*2)
    d3 = d_layer(d2, filters*4)
    d4 = d_layer(d3, filters*8)

    validity = SpectralNormalization(Conv2D(1, kernel_size=4, strides=1, padding='same'))(d4)

    return Model([img_A, img_B], validity)

def BuildGenerator(monoImageShape, colorChannelsCount, filters):
    """U-Net Generator"""
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.5)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same', activation='relu')(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.5)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=monoImageShape)

    # Downsampling
    d1 = conv2d(d0, filters, bn=False)
    d2 = conv2d(d1, filters*2)
    d3 = conv2d(d2, filters*4)
    d4 = conv2d(d3, filters*8)
    d5 = conv2d(d4, filters*8)
    d6 = conv2d(d5, filters*8)
    d7 = conv2d(d6, filters*8)

    # Upsampling
    u1 = deconv2d(d7, d6, filters*8)
    u2 = deconv2d(u1, d5, filters*8)
    u3 = deconv2d(u2, d4, filters*8)
    u4 = deconv2d(u3, d3, filters*4)
    u5 = deconv2d(u4, d2, filters*2)
    u6 = deconv2d(u5, d1, filters)

    u7 = UpSampling2D(size=2)(u6)
    output_img = SpectralNormalization(Conv2D(colorChannelsCount, kernel_size=4, strides=1, padding='same', activation='tanh'))(u7)

    return Model(d0, output_img)
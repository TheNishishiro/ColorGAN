# ColorGAN

GAN to color gray scaled anime images.

The goal was to keep the code as simple as possible so anyone can use it as a base for their own projects but it can also be ran as is.

## Getting Started

Below you will find instructions on how to run and train the ColorGAN model, I hope it helps and won't be too convoluted. 

### Prerequisites

To get things running you will fist have to install following:

```
Python 3.6 or newer
tensorflow 2.0 or newer
CUDA and cudNN compatible with your version of TF
Keras
numpy
glob
PIL
scipy
Flask (for API only)
```

### Installing and setup

Now onto installing.

First of all you will need a proper folder structure, most of it should be created automatically tho

```
ColorGAN 2
  dataset\
  test_output\
  test_pretrain_output\
  train_logs\
  trained_models\
```

And explanation for each one:

**dataset\\[datasetName]\\** - Here goes your colored dataset

**test_output** - Here you will find snapshots from your training progress

**test_pretrain_output** - Here you will find snapshots from your pretraining progress

**train_logs** - Here you will find training logs with your loss and accuracy

**trained_models** - Here you will find your trained models

## Preparing dataset

Put your colored images into **dataset\[datasetName]** folder

As of right now you need to make sure that your images are in JPG format and have the same size, but I want to make it more flexible in the future.

I used a dataset of about 90k 256x256 images of anime faces from [here](https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset). I cleaned it a little more before feeding it to the model since it still had some junk that weren't faces or didn't look much like human faces.

## Training

I used my RTX 3080 GPU for training and testing, with batch size of 16 to train it as quickly as possible.

First I trained the generator in a supervised manner for 10 epochs to bring it up to speed with the dataset (about 3 hours).

Then the adversarial training began where I trained both generator and discriminator for about 10 epochs to get presented results (about 6 hours)

Loss of a generator steadily went down and oscillated around 9-10

Discriminator was a pain tho, it was able to very quickly learn the difference between real and fake images with an accuracy close to 100%

For some reason reloading model mid training seems to help (?), not quite sure what's that about

## Using pretrained model

TODO

## Running the test

TODO

## Running web app

TODO

## Results

### Anime characters:

TODO

### Manga pages

TODO

## Models for download

TODO

## Dataset (18+ warning)

TODO

## License

This project is licensed under the MIT License

## Acknowledgments

* Pix2Pix implementation from https://github.com/eriklindernoren/Keras-GAN
* Dataset for faces from https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset

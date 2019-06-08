# HentaiGAN

GANs to color grayscaled manga pages

## Getting Started

Instructions below will help you set things up and get started so let's go!

### Prerequisites

To get things running you will fist have to install following:

```
Python 3.6
tensorflow (guess version doesn't really matter)
CUDA and cudNN compatible with your version of TF
Keras
matplotlib
numpy
glob
PIL
scipy
```

For training I used pictures downscaled to 256x384 with batch size of 1 and trained it on mobile GTX1050 4GB for approximately 2 hours. 

### Installing and setup

Now onto installing.

First of all you will need a proper folder structure

```
HentaiGAN
  data\
  images\
  input\
  output\
  models\
```

And explanation for each one:

**data\[datasetName]\\** - Here goes your colored datset

**images** - Here you will find snapshots from your training progress

**input** - Here you should put your test data and images you want to color

**output** - Colored images will be saved in this folder

**models** - A place for trained exported models

## Preparing dataset

Put your colored images in **data\[datasetName]** idealy these should be around the same aspect ratio.

Then you need to make some changes in code to personalize training:
```
        self.img_rows = 384 # Change this to your desired image height
        self.img_cols = 256 # Change this to your desired image width
```
Keep in mind that these have to be dividable by 64.

You can also change the batch size at the very bottom (default is one)
```
gan.train(epochs=40000, batch_size=1, sample_interval=200, model_name=model_name)
```

Reason for my values were dictated by a pretty weak GPU so I couldn't really afford better resolution nor bigger batch

## Using pretrained model

Just make sure these
```
        self.img_rows = 384 # Change this to your desired image height
        self.img_cols = 256 # Change this to your desired image width
```
are set to the same values as the pretrained model used because the same values are being used to load and rescale images, if you use different values input might have a wrong shape.

## Running the test

To run it, simply use following in your cmd or terminal:
```
python pix2pix.py
```
You'll be greeted with a simple text menu just follow instructions!

If you are using predict option keep in mind file you enter needs to be present in **input** folder.

## License

This project is licensed under the MIT License

## Acknowledgments

* Pix2Pix implementation from https://github.com/eriklindernoren/Keras-GAN
* Thank you to my friends for idea for this *beautiful* and *useful* project

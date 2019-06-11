# ColorGAN

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

For training I used upscaled 64x64 pictures to 256x256 with batch size of 16 and trained it on Google Colab for 10 epoches on 20,000 images
 
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

**data\\[datasetName]\\** - Here goes your colored datset

**images** - Here you will find snapshots from your training progress

**input** - Here you should put your test data and images you want to color

**output** - Colored images will be saved in this folder

**models** - A place for trained exported models

## Preparing dataset

Put your colored images in **data\[datasetName]** idealy these should be around the same aspect ratio.

Then you need to make some changes in code to personalize training:
```
        self.img_rows = 256 # Change this to your desired image height
        self.img_cols = 256 # Change this to your desired image width
```
Keep in mind that these have to be dividable by 64.

You can also change the batch size at the very bottom (default is one)
```
gan.train(epochs=200, batch_size=16, sample_interval=200, model_name=model_name)
```

Reason for my values were dictated by a pretty weak GPU so I couldn't really afford better resolution nor bigger batch

In case you want to train it with the same dataset you can download it [from here](https://github.com/Mckinsey666/Anime-Face-Dataset).

## Using pretrained model

Just make sure these
```
        self.img_rows = 256 # Change this to your desired image height
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

## Results

Training set:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/10_400.png)

Test set:
Epoch 0:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/0.png)
Epoch 1:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/1.png)
Epoch 2:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/2.png)
Epoch 3:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/3.png)
Epoch 4:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/4.png)
Epoch 5:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/5.png)
Epoch 6:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/6.png)
Epoch 7:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/7.png)
Epoch 8:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/8.png)
Epoch 9:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/9.png)
Epoch 10:
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/10.png)

## License

This project is licensed under the MIT License

## Acknowledgments

* Pix2Pix implementation from https://github.com/eriklindernoren/Keras-GAN
* Dataset from https://github.com/Mckinsey666/Anime-Face-Dataset
* Thank you to my friends for idea for this *beautiful* and *useful* project

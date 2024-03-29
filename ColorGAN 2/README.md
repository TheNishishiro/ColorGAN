# ColorGAN

GAN to color gray scaled anime images. It uses Pix2Pix GAN model for training

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
  test_output\
  test_pretrain_output\
  train_logs\
  trained_models\
  test_set\
  test_predictions\
```

And explanation for each one:

**test_output** - Here you will find snapshots from your training progress, inside of a folder with your model name

**test_pretrain_output** - Here you will find snapshots from your pretraining progress, inside of a folder with your model name

**train_logs** - Here you will find training logs with your loss and accuracy

**trained_models** - Here you will find your trained models, inside of a folder with your model name

**test_set** - Here goes your test image set for batch prediction

**test_predictions** - Here you will find your batch predictions

## Preparing dataset

As of right now you need to make sure that your images are in JPG format and have the same size, but I want to make it more flexible in the future.

I used a dataset of about 90k 256x256 images of anime faces from [here](https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset). I cleaned it a little more before feeding it to the model since it still had some junk that weren't faces or didn't look much like human faces.

### Converting dataset

There is a DatasetConverter.py in Converters folder that could help you in changing a style of your dataset.

For example you could convert every image in your dataset folder to a sketch or an oil painting, might come in handy with future models

## Training

I used my RTX 3080 GPU for training and testing, with batch size of 16 to train it as quickly as possible.

First I trained the generator in a supervised manner for 10 epochs to bring it up to speed with the dataset (about 3 hours).

Then the adversarial training began where I trained both generator and discriminator for about 11 epochs to get presented results (about 6 hours)

Loss of a generator steadily went down and oscillated around 9-10

Discriminator was a pain tho, it was able to very quickly learn the difference between real and fake images with an accuracy close to 100%

For some reason reloading model mid training seems to help (?), not quite sure what's that about

## Using pretrained model

In order to use my pretrained model put both *.h5 and *.json files into the **trained_models/[modelName]/** folder.

then run the application

```
python main.py
```

and follow on screen commands to load the model and predict an image:

```
1) Create new model
2) Load existing model
    2
    Enter model name: FaceGan
    Enter dataset directory: 
    Dataset convert style ("", "sketch"):
    Enter model version: 11
    Loading FaceGan_v11 model...
```
Dataset convert style can be used to convert dataset input style, for example empty will feed grayscaled images into the model while sketch style will convert images to sketch before feeding them in

from here you can resume training or predict images from **test_set** or any other folder

```
1) Full training
2) Pretrain generator
3) Train adversarial model
4) Predict batch
    4
    Enter gray scaled image directory:
    Save prediction director:
```

Empty prediction director will save predictions to **test_predictions** folder and empty grayscaled image directory will predict images from **test_set** folder

## Training model yourself

You can train the model yourself, just remember to adjust model input accordingly to image size you will be inputting

```
self.InputImageHeight = 256 # your image height
self.InputImageWidth = 256 # your image width
```

Keep in mind that image size needs to be divisible by 64 and if it smaller than 256x256 might require removing a few layers from the generator and discriminator

## Running web app

You can run the web app by running the following command:

```
python api.py ModelName ModelVersion

ex.
python api.py FaceGan 11
```

Endpoint is available at http://localhost:5000/pix2pix

so just send a POST request with an image to the endpoint and it will return the predicted image

This repository also contains a .NET application to POST images to this API and it is available [here](https://github.com/TheNishishiro/ColorGAN/tree/master/ColorGanInterface) 

## Results

Random samples, I didn't bother to search for the best or the worse

### Anime faces (Grayscale → Color):

Predictions for faces from dataset (random samples)

(1st row - input, 2nd row - output, 3rd row - ground truth)
![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_dataset.png)

Predictions for faces from outside of dataset

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_unseen.png)

### Anime faces (Sketch → Color)

This one was difficult to train and I had to pick a specific model snapshot for decent results, I'm not sure whether longer training would improve anything since it seemed to spit out trash every couple iterations

Dataset doesn't make it easier with messy backgrounds but I'm not sure how much of an impact it had in the end.

Anyways, I took the same approach as I did with grayscaled images, just that I converted the grayscaled images into sketches using OpenCV magic before feeding them into the network

Once again, results from the dataset (random samples)

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_sketch_dataset.png)

And predictions for unseen images

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_sketch.png)

### Manga pages

<details>
<summary>Hentai pages (+18)</summary>

This one is a stretch, fairly small dataset of about 30k pages without any filtering out of trash, pretrained generator for 20 epochs and trained adversarial model for another 20

In the end I think it did overfit as one issue, the other issue is that converting colored pages to grayscale does not reflect the same level of detail as original grayscaled manga which makes transfer much more difficult (e.g. accurate shading of objects)

results from the dataset (random samples)

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_hentai_dataset.png)

predictions for unseen images

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/ColorGAN%202/Example%20outputs/prediction_hentai_unseen.png)

I will be working on a new, cleaner dataset and look for a way to make colored manga look closer to grayscaled 

</details>

## Models for download

Model for faces **FaceGan** version **11** is avaiable [here](https://drive.google.com/file/d/1ISyhPghbO_PXITYofTanClbER1k6vNbk/view?usp=sharing)

Model for sketches **LineGan** version **10** is avaiable [here](https://drive.google.com/file/d/1rs3yWfHaelZOZJrUIGs1DCfQC2kjwJrN/view?usp=sharing)

Model for hentai **HentaiGan** version **40** is available [here](https://drive.google.com/file/d/1k96gUjq1Dnhn0gD0E6IN23FDSg2Dp80G/view?usp=sharing)

I'll post more models once I get to train them

## Dataset (18+ warning)

TODO, unavailable yet, check legacy implementation for old dataset

## License

This project is licensed under the MIT License

## Acknowledgments

* Pix2Pix implementation from https://github.com/eriklindernoren/Keras-GAN
* Dataset for faces from https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset

# ColorGAN

GANs to color grayscaled manga pages

## Getting Started

Instructions below will help you set things up and get started, just be aware that it's still early development, the code is a mess and a subject to changes which I'm still working on, but here we go anyway!

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
Flask (for web app only)
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
  static\output\  (for web app only)
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

(Download link to pretrained models can be found on the bottom)

Just make sure these
```
        self.img_rows = 256 # Change this to your desired image height
        self.img_cols = 256 # Change this to your desired image width
```
are set to the same values as the pretrained model used because the same values are being used to load and rescale images, if you use different values input might have a wrong shape.

Then put images in your **input** folder.

After running the program chose 2 from menu and insert model name as **face_v8** depending on your model

## Running the test

To run it, simply use following in your cmd or terminal:
```
python pix2pix_recon.py
```
You'll be greeted with a simple text menu just follow instructions!

If you are using predict option keep in mind file you enter needs to be present in **input** folder.

## Running web app

It's not quite ready, not even close, but if you're desperate to then here is how to get it to work

Edit line 7 and 15 of **main.py**
```
model = "face_v8"
UPLOAD_FOLDER = './input/'
```
to model which you want to use and a folder to which you want to save uploaded pics.

Then run
```
python main.py
```
Your server should go up on a localhost, for me 127.0.0.1:5000

Uploaded files will be save to upload folder, processed, saved into **static/output** and removed from upload folder.



No sunshines nor rainbows, it's just a simple thing I wrote for my presentation that haven't even been used but it's still there.

## Results

### Anime characters:

Training set:

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/10_400.png)

Test set:

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

### Manga pages

![alt text](https://raw.githubusercontent.com/TheNishishiro/ColorGAN/master/output/manga.png)

## Models for download

3 of my pretrained models can be found [here](https://drive.google.com/drive/folders/1e3vJHR9x4Yci6UAFbAz0rv14oh9HJwY3?usp=sharing)

Each model is represented by 4 files **d_name_version.json**, **d_name_version.h5**, **g_name_version.json** and **g_name_version.h5**.

*face* - For face coloring like examples up there (input of 256x256)

*hGAN* - For coloring of hentai pages (input of 256x384)(included versions 24 and 35 because I only checked v24 and am not sure if v35 isn't overfitted)

Download 4 files of a model and put them into **models** folder

## Dataset

Hentai dataset containing around 30k pages of colored hentai scaled down to 256x384 can be found [here](https://drive.google.com/file/d/1WkUn1CqaiPx9XD_V5X-s9VxybsdtgnwK/view?usp=sharing)

## License

This project is licensed under the MIT License

## Acknowledgments

* Pix2Pix implementation from https://github.com/eriklindernoren/Keras-GAN
* Dataset for faces from https://github.com/Mckinsey666/Anime-Face-Dataset
* Dataset for manga from nhentai (sorry for scrapping ♥)
* Thank you to my friends for idea for this *beautiful* and *useful* project

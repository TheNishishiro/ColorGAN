from PIL import Image, ImageCms
from random import shuffle
from glob import glob
import numpy as np
import os

def LoadBatch(batchSize, datasetFolder, size, isResize = True, isFlipLeftRight = True, isFlipTopDown = False):
    path = glob(f'{datasetFolder}/*')
    shuffle(path)
    numberOfBatches = int(len(path) / batchSize)
    for i in range(numberOfBatches - 1):
        batch = path[i * batchSize:(i+1)*batchSize]
        coloredImages, monoImages = [], []
        
        for image in batch:
            leftToRightFlipSeed = np.random.random()
            upsideDownFlipSeed = np.random.random()
            imageMono = LoadAndTransformImage(image, 'L', size, leftToRightFlipSeed, upsideDownFlipSeed, isResize, isFlipLeftRight, isFlipTopDown)
            imageColor = LoadAndTransformImage(image, 'RGB', size, leftToRightFlipSeed, upsideDownFlipSeed, isResize, isFlipLeftRight, isFlipTopDown)
            
            imageMono = np.array(imageMono)
            imageColor = np.array(imageColor)
            
            coloredImages.append(imageColor)
            monoImages.append(imageMono)
            
        yield NormalizeColorValues(coloredImages), NormalizeColorValues(monoImages), numberOfBatches

def LoadImage(imagePath, size, isResize = True):
    img = []
    img2 = []
    leftToRightFlipSeed = np.random.random()
    upsideDownFlipSeed = np.random.random()
    monoImage = LoadAndTransformImage(imagePath, 'L', size, leftToRightFlipSeed, upsideDownFlipSeed, isResize, False, False)
    colorImage = LoadAndTransformImage(imagePath, 'RGB', size, leftToRightFlipSeed, upsideDownFlipSeed, isResize, False, False)
    
    monoImage = NormalizeColorValues(monoImage)
    img.append(monoImage)
    img = np.array(img)
    
    colorImage = NormalizeColorValues(colorImage)
    img2.append(colorImage)
    img2 = np.array(img2)
    
    return [img, img2]

def LoadAndTransformImage(image, convertType, size, leftFlipSeed, upsideDownFlipSeed, isResize = True, isFlipLeftRight = True, isFlipTopDown = False):
    loadedImage = Image.open(image).convert(convertType)
    if isResize == True:
        loadedImage = loadedImage.resize(size, Image.ANTIALIAS)
    if leftFlipSeed > 0.5 and isFlipLeftRight == True:
        loadedImage = loadedImage.transpose(Image.FLIP_LEFT_RIGHT)
    if upsideDownFlipSeed > 0.5 and isFlipTopDown == True:
        loadedImage = loadedImage.transpose(Image.FLIP_TOP_BOTTOM)
    return loadedImage

def NormalizeColorValues(image):
    return (np.array(image) / 127.5) - 1

def UnNormalizeColorValues(image):
    return (0.5 * np.array(image) + 0.5) * 255

def Predict(generator, folderPath, outputFolder, epoch, batchIndex, size, randomize=True):
    os.makedirs(folderPath, exist_ok=True)
    path = glob(f'{folderPath}/*')
    
    if randomize == True:
        batch_images = np.random.choice(path, size=3)
    else:
        batch_images = path

    conc = []
    conFinal = []
    for inp in batch_images:

        fake_A = np.expand_dims(LoadImage(inp, size)[0], axis=3)
        fake_A = generator.predict(fake_A)
        fake_A = (fake_A + 1)*127.5

        gray = LoadImage(inp, size)[0]
        gray = np.stack((gray,)*3, axis=-1)
        gray = (gray + 1)*127.5

        col = LoadImage(inp, size)[1]
        col = (col + 1)*127.5

        conc.append(np.vstack((gray[0], fake_A[0], col[0])))

        if(len(conc) == 2 and conFinal == []):
            conFinal = np.hstack((conc[0], conc[1]))
            conc = []
        elif(len(conc) == 1 and conFinal != []):
            conFinal = np.hstack((conc[0], conFinal))
            conc = []

    image = Image.fromarray(conFinal.astype('uint8'), 'RGB')
    os.makedirs(outputFolder, exist_ok=True)
    if epoch == None:
        image.save(f"{outputFolder}/prediction.png")
    else:
        image.save(f"{outputFolder}/{epoch}_{batchIndex}.png")
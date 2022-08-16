from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import tensorflow.keras as K

import datetime
import os
import numpy as np
from PIL import Image

from Builders.ModelBuilder import BuildDiscriminator, BuildGenerator
from DataLoaders.FolderDatasetLoader import LoadBatch, Predict, LoadImage, UnNormalizeColorValues, NormalizeColorValues

ModelSavePath = "./trained_models/"
LogPath = "./train_logs/"

class Pix2Pix():
    def __init__(self, modelName, datasetFolder, inputConvertStyle):
        self.ModelName = modelName
        self.DataSetFolder = datasetFolder
        self.MonoStyle = inputConvertStyle
        self.ColorStyle = None
        self.InputImageHeight = 256
        self.InputImageWidth = 256
        self.EpochStart = 0
        
        self.ColorChannelsCount = 3
        self.ColorImageShape = (self.InputImageHeight, self.InputImageWidth, self.ColorChannelsCount)
        
        self.MonoChannelsCount = 1
        self.MonoImageShape = (self.InputImageHeight, self.InputImageWidth, self.MonoChannelsCount)
        
        # Calculate output shape of D (PatchGAN)
        self.PatchHeight = int(self.InputImageHeight / 2**4)
        self.PatchWidth = int(self.InputImageWidth / 2**4)
        self.DiscriminatorPatch = (self.PatchHeight, self.PatchWidth, 1)
        
        # Number of filters in the first layer of G and D
        self.GeneratorInputFiltersCount = 64
        self.DiscriminatorInputFiltersCount = 64
        
        self.GeneratorOptimizer = Adam(0.0002, 0.5)
        self.DiscriminatorOptimizer = Adam(0.0002, 0.5)
        
    def Summary(self):
        print("DISCRIMINATOR MODEL:")
        self.Discriminator.summary()
        print("GENERATOR MODEL:")
        self.Generator.summary()
        print("COMBINED MODEL:")
        self.Pix2PixModel.summary()
        
    def Create(self):
        print(f"Constructing {self.ModelName} model...")
        self.Discriminator = BuildDiscriminator(self.ColorImageShape, self.MonoImageShape, self.DiscriminatorInputFiltersCount)
        self.Generator = BuildGenerator(self.MonoImageShape, self.ColorChannelsCount, self.GeneratorInputFiltersCount)
        self.ConstructCombined()
        
    def Load(self, version):
        self.EpochStart = version
        print(f"Loading {self.ModelName}_v{version} model...")
        jsonFile = open(f"{ModelSavePath}{self.ModelName}/d_{self.ModelName}_v{version}.json", 'r')
        loadedJsonModel = jsonFile.read()
        jsonFile.close()
        self.Discriminator = model_from_json(loadedJsonModel)
        self.Discriminator.load_weights(f"{ModelSavePath}{self.ModelName}/d_{self.ModelName}_v{version}.h5", 'r')
        print("Discriminator loaded successfuly")
        
        jsonFile = open(f"{ModelSavePath}{self.ModelName}/g_{self.ModelName}_v{version}.json", 'r')
        loadedJsonModel = jsonFile.read()
        jsonFile.close()
        self.Generator = model_from_json(loadedJsonModel)
        self.Generator.load_weights(f"{ModelSavePath}{self.ModelName}/g_{self.ModelName}_v{version}.h5", 'r')
        print("Generator loaded successfuly")
        
        self.ConstructCombined()
        
    def Save(self, epoch):
        os.makedirs(ModelSavePath, exist_ok=True)
        modelName = self.ModelName + "_v" + str(epoch)
        self.SaveModel(self.Discriminator, f"d_{modelName}")
        self.SaveModel(self.Generator, f"g_{modelName}")
        
    def SaveModel(self, model, modelFullName):
        os.makedirs(f'{ModelSavePath}{self.ModelName}', exist_ok=True)
        jsonModel = model.to_json()
        with open(f"{ModelSavePath}{self.ModelName}/{modelFullName}.json", "w") as jsonFile:
            jsonFile.write(jsonModel)
        model.save_weights(f"{ModelSavePath}{self.ModelName}/{modelFullName}.h5")
        
    def ConstructCombined(self):
        self.Discriminator.compile(loss='mse', optimizer=self.DiscriminatorOptimizer, metrics=['accuracy'])
        self.Generator.compile(loss='mae', optimizer=self.GeneratorOptimizer, metrics=["accuracy"])
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------
        
        # Input images and their conditioning images
        colorImage = Input(shape=self.ColorImageShape)
        monoImage = Input(shape=self.MonoImageShape)
        
        # By conditioning on B generate a fake version of A
        generateColoredImage = self.Generator(monoImage)
        
        # For the combined model we will only train the generator
        self.Discriminator.trainable = False
        
        # Discriminators determines validity of translated images / condition pairs
        isValid = self.Discriminator([generateColoredImage, monoImage])
        
        self.Pix2PixModel = Model(inputs=[colorImage, monoImage], outputs=[isValid, generateColoredImage])
        self.Pix2PixModel.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=self.GeneratorOptimizer)
        
    def Train(self, epochs, batchSize=1, sampleInterval=50):
        start_time = datetime.datetime.now()
        log = open(f'{LogPath}training_{self.ModelName}_{start_time.strftime("%Y_%m_%d-%I_%M_%S_%p")}.log', 'w')
        
        # Adversarial loss ground truths
        valid = np.ones((batchSize,) + self.DiscriminatorPatch) * 0.9
        fake = np.zeros((batchSize,) + self.DiscriminatorPatch)
        for epoch in range(self.EpochStart, epochs):
            for batch_i, (colorImages, monoImages, batches) in enumerate(LoadBatch(batchSize, self.DataSetFolder, (self.InputImageWidth, self.InputImageHeight), False, True, False, self.MonoStyle, self.ColorStyle)):
                # Color image
                fakeColor = self.Generator.predict(monoImages)
                
                # Train discriminator to distinguish fakes from reals
                discriminatorLossReal = self.Discriminator.train_on_batch([colorImages, monoImages], valid)
                discriminatorLossFake = self.Discriminator.train_on_batch([fakeColor, monoImages], fake)
                discriminatorLoss = 0.5 * np.add(discriminatorLossReal, discriminatorLossFake)
                
                # Train generator
                generatorLoss = self.Pix2PixModel.train_on_batch([colorImages, monoImages], [valid, colorImages])
                
                elapsed_time = datetime.datetime.now() - start_time
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,batch_i, batches, discriminatorLoss[0], 100*discriminatorLoss[1], generatorLoss[0], elapsed_time))
                log.write(f'\n{epoch};{batch_i};{discriminatorLoss[0]};{100*discriminatorLoss[1]};{generatorLoss[0]};{elapsed_time}')
                
                if batch_i % sampleInterval == 0:
                    self.Save(str(epoch))  
                    Predict(self.Generator, self.DataSetFolder, f"./test_output/{self.ModelName}", str(epoch), str(batch_i), (self.InputImageWidth, self.InputImageHeight), True, self.MonoStyle, self.ColorStyle)
                
    def PreTrainGenerator(self, epochs, batchSize=1, sampleInterval=50):
        os.makedirs(LogPath, exist_ok=True)
        start_time = datetime.datetime.now()
        log = open(f'{LogPath}pretrain_{self.ModelName}_{start_time.strftime("%Y_%m_%d-%I_%M_%S_%p")}.log', 'w')
        
        for epoch in range(self.EpochStart, epochs):
            for batch_i, (colorImages, monoImages, batches) in enumerate(LoadBatch(batchSize, self.DataSetFolder, (self.InputImageWidth, self.InputImageHeight), True, True, True, self.MonoStyle, self.ColorStyle)):
                
                gLoss = self.Generator.train_on_batch(monoImages, colorImages)
                elapsed_time = datetime.datetime.now() - start_time
                
                print ("[Epoch %d/%d] [Batch %d/%d] [G loss: %f/%f] time: %s" % (epoch, epochs, batch_i, batches, gLoss[0], gLoss[1], elapsed_time))
                log.write(f'\n{epoch};{batch_i};{gLoss[0]};{gLoss[1]};{elapsed_time}')
                if batch_i % sampleInterval == 0:
                    self.Save(str(epoch))  
                    Predict(self.Generator, self.DataSetFolder, f"./test_pretrain_output/{self.ModelName}", str(epoch), str(batch_i), (self.InputImageWidth, self.InputImageHeight), True, self.MonoStyle, self.ColorStyle)
                    
    def PredictFromPath(self, imagePath, savePath = None):
        image = LoadImage(imagePath, (self.InputImageHeight, self.InputImageWidth), True)
        
        input = np.expand_dims(image[0], axis=3)
        prediction = self.Generator.predict(input)
        prediction = UnNormalizeColorValues(prediction)
        
        image = Image.fromarray(prediction[0].astype('uint8'), 'RGB')
        if savePath != None:
            image.save(f"{savePath}/prediction.png")
        return image
    
    def PredictFromImage(self, image, savePath = None):
        imageNormalize = NormalizeColorValues(image)
        
        prediction = self.Generator.predict(np.expand_dims([imageNormalize], axis=3))
        prediction = UnNormalizeColorValues(prediction)
        
        image = Image.fromarray(prediction[0].astype('uint8'), 'RGB')
        if savePath != None:
            image.save(f"{savePath}/prediction.png")
        return image
    
    def PredictBatchFromPatch(self, imagesPath, savePath = None):
        Predict(self.Generator, imagesPath or './test_set', savePath or './test_predictions', None, None, (self.InputImageWidth, self.InputImageHeight), False, self.MonoStyle, self.ColorStyle, 8)
        



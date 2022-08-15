from Model.pix2pix import Pix2Pix

gan = Pix2Pix("FaceGan", "D:\\Datasets\\animefaces256cleaner")
gan.Load(9)
gan.Summary()
#gan.PreTrainGenerator(10, 16, 200)
gan.Train(2000, 16, 200)
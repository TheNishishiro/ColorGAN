from Model.pix2pix import Pix2Pix

if __name__ == '__main__':
    option = input('1) Create new model\n2) Load existing model\n')
    gan = Pix2Pix(input('Enter model name: '), input('Enter dataset directory: '), input('Dataset convert style ("", "sketch"): '))
    
    if option == '1':
        gan.Create()
    elif option == '2':
        gan.Load(int(input('Enter model version: ')))
    gan.Summary()
    
    option = input('1) Full training\n2) Pretrain generator\n3) Train adversarial model\n4) Predict batch\n')
    
    if option == '1' or option == '2' or option == '3':
        batchSize = int(input('Enter batch size: '))
        epochsCount = int(input('Number of epochs (will be applied to pretrain if checked): '))
        snapshotTime = int(input('Snapshot every x batches: '))
    
    if option == '1':
        adversarialEpochCount = int(input('Number of epochs for adversarial training: '))
        gan.PreTrainGenerator(epochsCount, batchSize, snapshotTime)
        gan.Train(adversarialEpochCount, batchSize, snapshotTime)
    if option == '2':
        gan.PreTrainGenerator(epochsCount, batchSize, snapshotTime)
    if option == '3':
        gan.Train(epochsCount, batchSize, snapshotTime)
    if option == '4':
        gan.PredictBatchFromPatch(input('Enter gray scaled image directory: '), input('Save prediction director: '))
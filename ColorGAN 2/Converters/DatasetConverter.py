import cv2
from glob import glob
import os
from PIL import Image
import numpy as np

def PilToOpenCv(pilImage):
    numpy_image=np.array(pilImage)  
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

def OpenCvToPil(cvImage):
    color_coverted = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)

def ConvertToLineArt(datasetPath, savePath):
    path = glob(f'{datasetPath}/*')
    os.makedirs(savePath, exist_ok=True)
    i = 0
    for img in path:
        fileName = os.path.basename(img)
        print(f'[{i}/{len(path)}]Converting {fileName}...')
        
        image = cv2.imread(img)
        sketch = CvImageToLineArt(image)
        cv2.imwrite(f'{savePath}/{fileName}', sketch)
        

def CvImageToLineArt(cvImage):
    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21,21), 0)
    invertedB = 255 - blurred
    return cv2.divide(gray, invertedB, scale=256.0)

def ConvertToOil(datasetPath, savePath):
    path = glob(f'{datasetPath}/*')
    os.makedirs(savePath, exist_ok=True)
    i = 0
    for img in path:
        fileName = os.path.basename(img)
        print(f'[{i}/{len(path)}]Converting {fileName}...')
        
        image = cv2.imread(img)
        oil = CvImageToOil(image)
        cv2.imwrite(f'{savePath}/{fileName}', oil)
        

def CvImageToOil(cvImage):
    return cv2.xphoto.oilPainting(src=cvImage, size=8, dynRatio=1)

if __name__ == '__main__':
    options = input('1) Convert to lineart\n2) Convert to oil painting\n')
    sourceDir = input('Source dataset directory: ')
    outputDir = input('Output dataset directory: ')
    
    if options == '1':
        ConvertToLineArt(sourceDir, outputDir)
    elif options == '2':
        ConvertToOil(sourceDir, outputDir)

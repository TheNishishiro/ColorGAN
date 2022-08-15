import cv2
from glob import glob
import os

def ConvertToLineArt(datasetPath, savePath):
    path = glob(f'{datasetPath}/*')
    os.makedirs(savePath, exist_ok=True)
    i = 0
    for img in path:
        fileName = os.path.basename(img)
        print(f'[{i}/{len(path)}]Converting {fileName}...')
        
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21,21), 0)
        invertedB = 255 - blurred
        sketch = cv2.divide(gray, invertedB, scale=256.0)
        cv2.imwrite(f'{savePath}/{fileName}', sketch)
        
def ConvertToOil(datasetPath, savePath):
    path = glob(f'{datasetPath}/*')
    os.makedirs(savePath, exist_ok=True)
    i = 0
    for img in path:
        fileName = os.path.basename(img)
        print(f'[{i}/{len(path)}]Converting {fileName}...')
        
        image = cv2.imread(img)
        oil = cv2.xphoto.oilPainting(src=image, size=8, dynRatio=1)
        cv2.imwrite(f'{savePath}/{fileName}', oil)
        
        
if __name__ == '__main__':
    options = input('1) Convert to lineart\n2) Convert to oil painting\n')
    sourceDir = input('Source dataset directory: ')
    outputDir = input('Output dataset directory: ')
    
    if options == '1':
        ConvertToLineArt(sourceDir, outputDir)
    elif options == '2':
        ConvertToOil(sourceDir, outputDir)

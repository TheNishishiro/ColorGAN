import numpy as np
from PIL import Image
from io import BytesIO

def Colorize(p2pModel, file):
    img = Image.open(file).convert('L')
    img = p2pModel.PredictFromImage(img)
    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return img_io
from PIL import Image
import os

# def prepareImages(path):
#     images = [os.join(path, file) for file in os.listdir(path)]

prepareImages = lambda path : [os.path.join(path, file) for file in os.listdir(path)]
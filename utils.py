from PIL import Image
import os

# def prepareImages(path):
#     images = [os.join(path, file) for file in os.listdir(path)]

prepareImages = lambda path : [os.path.join(path, file) for file in os.listdir(path)]

def tweak_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


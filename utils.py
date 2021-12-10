from PIL import Image
import os


prepareImages = lambda path : [os.path.join(path, file) for file in os.listdir(path)]

def tweak_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image

from utils import prepareImages
class UnpairedTrainDataset(data.Dataset):
    def __init__(self):
        self.A_path = 'maps/trainA'
        self.B_path = 'maps/trainB'

        self.A_imgpath = prepareImages(self.A_path)
        self.B_imgpath = prepareImages(self.B_path)

        self.A_size = len(self.A_imgpath)
        self.B_size = len(self.B_imgpath)

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        A_instance_path = self.A_imgpath[idx]
        B_instance_path = self.B_imgpath[torch.randint(self.B_size, (1,))]

        A = Image.open(A_instance_path)
        B = Image.open(B_instance_path)

        A_tensor = self.transform(A)
        B_tensor = self.transform(B)

        # clever idea of storing in dictionary is borrowed from official cyclegan implementation #
        d = dict()
        d['A'] = A_tensor
        d['B'] = B_tensor
        d['A_path'] = A_instance_path
        d['B_path'] = B_instance_path

        return d
    
    def __len__(self):
        return max(self.A_size, self.B_size)

if __name__ == '__main__':
    dataset = UnpairedTrainDataset()

import torch
from torch.utils.data import DataLoader
from dataset import UnpairedTrainDataset

dataset = UnpairedTrainDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for i, data in enumerate(loader):
    print(data['B'].size())
    break
import torch
from torch.utils.data import DataLoader
from dataset import UnpairedTrainDataset
from gan import CycleGAN
from gen_disc import Generator
import os
import random
from PIL import Image
from torchvision import transforms

MODE = 'train'

if MODE == 'train':
    dataset = UnpairedTrainDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = CycleGAN()
    for epoch in range(50):
        print('-'*50)
        print(f'We are at epoch {epoch}')
        print('-'*50)
        for i, data in enumerate(loader):
            model.train_one_epoch(data)
    torch.save(model.AB_gen.state_dict(), 'ab.pth')
    torch.save(model.BA_gen.state_dict(), 'ba.pth')

elif MODE=='test':
    model_AB = Generator()
    model_BA = Generator()
    model_AB.load_state_dict(torch.load('ab.pth'))
    model_BA.load_state_dict(torch.load('ba.pth'))
    ran = random.randint(1, 1096) # 1096 is image count in test image directory
    img_pathA = f'./maps/testA/{ran}_A.jpg'
    img_pathB = f'./maps/testB/{ran}_B.jpg'
    imgA = Image.open(img_pathA)
    imgB = Image.open(img_pathB)
    img_tensor_A = transforms.ToTensor()(imgA)
    img_tensor_B = transforms.ToTensor()(imgB)

    # d = {'A': img_tensor_A, 'B': img_tensor_B, 'A_path': img_pathA, 'B_path': img_pathB}
    fakeA = model_AB(img_tensor_A.unsqueeze(0))
    assert(fakeA.shape == torch.Size([1, 3, 600, 600]))
    im = transforms.ToPILImage()(fakeA[0]).convert("RGB")
    im.show()


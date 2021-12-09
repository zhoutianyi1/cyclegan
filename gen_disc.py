import torch
import torch.nn as nn

class Generator:
    def __init__(self):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, super).__init__()
        kernel_size = 4
        input_c = 3
        pad = 1
        sequence = [nn.Conv2d(input_c, 1, kernel_size=kernel_size, stride=2, padding=pad), nn.LeakyReLU(0.2, True)]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(nf_mult_prev, nf_mult, kernel_size=kernel_size, stride=2, padding=pad),
                nn.BatchNorm2d(nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**3 , 8)
        sequence += [
            nn.Conv2d(nf_mult_prev, nf_mult, kernel_size=kernel_size, stride=2, padding=pad),
            nn.BatchNorm2d(nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(nf_mult, 1, kernel_size=kernel_size, stride=1, padding=pad)]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)


    
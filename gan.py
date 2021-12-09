import torch
import torch.nn as nn
from itertools import chain
from gen_disc import Generator, Discriminator

from utils import tweak_grad

device = 'cuda'
class CycleGAN:
    def __init__(self, lambda_cycle = 2):
        self.lambda_cycle = lambda_cycle
        self.AB_gen = Generator().to(device)
        self.BA_gen = Generator().to(device)
        self.A_disc = Discriminator().to(device)
        self.B_disc = Discriminator().to(device)

        self.gan_loss = nn.BCEWithLogitsLoss()
        self.cycle_loss = nn.L1Loss()
        self.gen_optimizer = torch.optim.Adam(chain(self.AB_gen.parameters(), self.BA_gen.parameters()), lr=0.005)
        self.disc_optimizer = torch.optim.Adam(chain(self.A_disc.parameters(), self.B_disc.parameters()), lr= 0.005)
    
    def forward(self, batch):
        As, Bs = batch['A'], batch['B']
        fakeBs = self.AB_gen(As)
        fakeAs = self.BA_gen(Bs)
        reconstruct_As = self.BA_gen(fakeBs)
        reconstruct_Bs = self.AB_gen(fakeAs)
        return { 'realAs': As, 'realBs': Bs, 'fakeAs': fakeAs, 'fakeBs': fakeBs, 'reconstruct_As': reconstruct_As, 'reconstruct_Bs': reconstruct_Bs }
        
    
    def _get_gen_loss(self, fake, net):
        y_fake = net(fake)
        loss = self.gan_loss(y_fake, torch.ones_like(y_fake))
        return loss

    def gen_backward(self, data):
        fakeAs, fakeBs, reconstruct_As, reconstruct_Bs = data['fakeAs'], data['fakeBs'], data['reconstruct_As'], data['reconstruct_Bs']
        realAs, realBs = data['realAs'], data['realBs']
        cycle_loss = self.lambda_cycle * (self.cycle_loss(realAs, reconstruct_As) + self.cycle_loss(realBs, reconstruct_Bs))
        gan_loss_A = self._get_gen_loss(fakeAs, self.A_disc)
        gan_loss_B = self._get_gen_loss(fakeBs, self.B_disc)
        gen_loss =  cycle_loss + gan_loss_A + gan_loss_B
        gen_loss.backward()

    def _get_disc_loss(self, fake, real, net):
        pred_real = net(real)
        loss_real = self.gan_loss(pred_real, torch.ones_like(pred_real))

        pred_fake= net(fake)
        loss_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))

        disc_loss = (loss_real + loss_fake) / 2
        disc_loss.backward()

    def disc_backward(self, data):
        fakeAs, fakeBs = data['fakeAs'], data['fakeBs']
        realAs, realBs = data['realAs'], data['realBs']
        self._get_disc_loss(fakeAs, realAs, self.A_disc)
        self._get_disc_loss(fakeBs, realBs, self.B_disc)


    def train_one_epoch(self, batch):
        data = self.forward(batch)
        tweak_grad([self.A_disc, self.B_disc])
        self.gen_optimizer.zero_grad()
        self.gen_backward(data)
        self.gen_optimizer.step()

        tweak_grad([self.A_disc, self.B_disc], True)
        self.disc_optimizer.zero_grad()
        self.disc_backward(data)
        self.disc_optimizer.step()
        
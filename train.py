import math

import torch
import torch.nn.functional as F
from matplotlib import gridspec
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import sampler
from torchvision import datasets, transforms
import os
from models import *
import numpy as np

dtype = torch.FloatTensor
IMG_DIM=28
NOISE_DIM=50

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def g_bceloss(scores_fake):
    return nn.BCEWithLogitsLoss()(scores_fake, torch.ones(scores_fake.shape[0]).float().unsqueeze(-1))

def d_bceloss(D, realx, fakex, y):
    N = realx.shape[0]
    real_scores = D(realx, y)
    fake_scores = D(fakex, y)
    return nn.BCEWithLogitsLoss()(real_scores, torch.ones(N).float().unsqueeze(-1)) \
        + nn.BCEWithLogitsLoss()(fake_scores, torch.zeros(N).float().unsqueeze(-1))

def noise(batch_size, noise_dim):
    return torch.rand((batch_size, noise_dim)) * 2 - 1


class GAN:

    def __init__(self, generator, discriminator, d_loss, g_loss, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.num_epochs = 10
        self.every = 100
        self.d_overtrain = kwargs.get("d_overtrain", 1)
        self.batch_size = kwargs.get("batch_size", 128)
        self.img_size = kwargs.get("img_size", 28)
        self.noise_dim = kwargs.get("noise_dim", 50)
        self.n_classes = kwargs.get("n_classes", None)
        self.d_optim = kwargs.get("d_optim",
                                  lambda params: torch.optim.Adam(params, lr=0.0002, betas=(0.5, 1 - 1e-3)))(
            self.discriminator.parameters())
        self.g_optim = kwargs.get("g_optim",
                                  lambda params: torch.optim.Adam(params, lr=0.0002, betas=(0.5,
                                                                                                                 1 -
                                                                                                                 1e-3)))(self.generator.parameters())
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.load_data()


    def load_data(self):
        transform = transforms.Compose(
            [
                # transforms.Resize(20),
                transforms.ToTensor(),
             ])
        dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset1, batch_size=self.batch_size)


    def sample_generator(self, model_path):
        net = torch.load(model_path)
        net.eval()
        N = 49
        labels = (torch.ones(7,7)*(torch.arange(7).unsqueeze(-1))).type(torch.long)
        with torch.no_grad():
            fake_imgs = net(noise(N,NOISE_DIM), labels.view(-1))
        self.show_images(fake_imgs.squeeze(), "samples")


    def train(self, name):
        if not hasattr('self', 'train_loader'):
            print("loading data")
            self.load_data()
        print("Starting training")
        counter = 0
        for epoch in range(self.num_epochs):
            for x, y in self.train_loader:
                if len(x) != self.batch_size:
                    continue
                for _ in range(self.d_overtrain):
                    d_loss, fake_images = self.train_discriminator(x,y)
                g_loss = self.train_generator()
                if counter % self.every == self.every - 1:
                    self.log_status(counter, d_loss, g_loss, fake_images, name)
                counter += 1


    def train_discriminator(self, realx, realy):
        # train on real images
        self.discriminator.zero_grad()
        realx = 2 * (realx - 0.5)
        # train on fake images
        with torch.no_grad():
            z = noise(self.batch_size, self.noise_dim)
            fakex = self.generator(z, realy)  # (batch_size, noise_dim)
        loss = self.d_loss(self.discriminator, realx, fakex, realy)
        loss.backward()
        self.d_optim.step()

        return loss, fakex


    def train_generator(self):
        self.generator.zero_grad()
        fakey = torch.LongTensor(np.random.randint(0, self.n_classes, self.batch_size))
        z = noise(self.batch_size, self.noise_dim)
        fake_images = self.generator(z, fakey)
        d_scores = self.discriminator(fake_images, fakey)
        g_loss = self.g_loss(d_scores)
        g_loss.backward()
        self.g_optim.step()
        return g_loss


    def log_status(self, counter, d_loss, g_loss, fake_images, name):
        print(f"Epoch {counter}: D loss: {d_loss}  g loss: {g_loss}")
        fake_images = fake_images.data.cpu().numpy().reshape(-1, self.img_size, self.img_size)
        self.show_images(fake_images, f"{name}_imgs_step_{counter}.png")
        torch.save(self.discriminator, f"models/discriminator_{counter}")
        torch.save(self.generator, f"models/generator_{counter}")


    def show_images(self, images,filename=None, max_imgs=50):
        # images = images[:max_imgs].view(-1,28,28)
        gridlen = math.floor(math.sqrt(max_imgs))
        fig = plt.figure(figsize=(gridlen, gridlen))
        gs = gridspec.GridSpec(gridlen, gridlen)
        gs.update(wspace=0.05, hspace=0.05)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img)
            if i == gridlen ** 2 - 1:
                break
        if filename:
            plt.savefig(os.path.join("fake_imgs", filename))
        else:
            plt.show()


if __name__ == "__main__":
    g_params = {"noise_dim": NOISE_DIM,
                "img_dim": IMG_DIM,
                "hidden_dim": 1024,
                "num_conv1": 128,
                "size_conv1": 4,
                "num_conv2": 64,
                "size_conv2": 4,
                "n_classes": 10,
                "label_emb_dim": 50
                }
    d_params = {"img_dim": IMG_DIM,
                "num_conv1": 32,
                "size_conv1": 5,
                "num_conv2": 64,
                "size_conv2": 5,
                "n_classes": 10,
                "label_emb_dim": 50
                }
    train_params = {"noise_dim": NOISE_DIM,
                    "img_size": IMG_DIM,
                    "batch_size": 128,
                    "d_overtrain": 3,
                    "n_classes": 10,
                    "d_loss": d_bceloss,
                    "g_loss": g_bceloss
                    }

    G = cDCGenerator(**g_params)
    D = cDCDiscriminator(**d_params)

    gan = GAN(G, D, **train_params)
    # gan.train("cGAN")
    gan.sample_generator("models/generator_4299")

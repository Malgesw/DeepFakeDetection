import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import wandb
import plotly.express as px

'''
code inspired by julianstastny's implementation (https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py)
'''


def loss_function(x, x_prime, mean, log_var):
    kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return kl_divergence + F.mse_loss(x, x_prime, reduction='sum')


#def reconstruction_score(x, x_prime):
 #   return torch.sqrt(F.mse_loss(x, x_prime))


class ResizedConv2d(nn.Module):  # Convolutional 2d block preceded by an interpolation operation to perform upsampling

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x


class ResidualBlockDec(nn.Module):

    def __init__(self, in_channels, stride=1):
        super().__init__()

        channels = int(in_channels/stride)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.residual = nn.Sequential()
        else:
            self.conv1 = ResizedConv2d(in_channels, channels, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(channels)
            self.residual = nn.Sequential(
                ResizedConv2d(in_channels, channels, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.residual(x)
        out = torch.relu(out)
        return out


class Resnet18Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc2 = nn.Linear(latent_dim, 512)
        self.res4 = self._create_residual_layer(512, 2, 2)
        self.res3 = self._create_residual_layer(256, 2, 2)
        self.res2 = self._create_residual_layer(128, 2, 2)
        self.res1 = self._create_residual_layer(64, 2, 1)
        self.conv1 = ResizedConv2d(64, 64, 3, 2)  # max pooling layer
        self.conv2 = nn.Conv2d(64, 64, 7, 2, 1)

    def _create_residual_layer(self, in_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in reversed(strides):
            layers += ResidualBlockDec(in_channels, strd)
        return nn.Sequential(*layers)  # "unfolds" the list of layers

    def forward(self, z):
        x = self.fc2(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.res4(x)
        x = self.res3(x)
        x = self.res2(x)
        x = self.res1(x)
        x = self.conv1(x)
        x = torch.sigmoid(self.conv2(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x


class Resnet18VAE(nn.Module):

    def __init__(self, device, latent_dim=256, hidden_dim=1024):
        super(Resnet18VAE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(*list(models.resnet18(weights='IMAGENET1K_V1').children())[-1])  # remove original fc layer
        self.hidden_fc1 = nn.Linear(512, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # batch normalization of a fc layer before passing x to mean and logvar layers

        self.mean_layer = nn.Linear(512, latent_dim)
        self.log_var_layer = nn.Linear(512, latent_dim)

        self.decoder = Resnet18Decoder(latent_dim=latent_dim)

    def reparametrize(self, mean, dev_std):
        epsilon = torch.rand_like(dev_std).to(self.device)  # normal standard random variable
        return dev_std * epsilon + mean  # z = std_dev(x)*epsilon + mean(x)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.relu(self.bn1(self.hidden_fc1(x)))
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        return mean, log_var

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparametrize(mean, log_var)
        x_prime = self.decode(z)
        return x_prime, mean, log_var

    def train_model(self, tr_loader, v_loader, t_loader, optim, num_epochs, use_test=False):

        if use_test:
            loader = t_loader
            desc = 'testing the model...'
            output_print = ("Epoch [{}/{}], train loss: {:.4f}, train reconstruction score: {:.4f}, test loss: {:.4f},"
                            " test reconstruction score: {:.4f}")
        else:
            loader = v_loader
            desc = 'validating the model...'
            output_print = ("Epoch [{}/{}], train loss: {:.4f}, train reconstruction score: {:.4f}, val loss: {:.4f},"
                            " val reconstruction score: {:.4f}")

        for epoch in range(num_epochs):

            self.train()
            train_loss = 0.0
            train_acc = 0.0

            for batch in tqdm(tr_loader, desc='training the model...'):

                batch = batch.to(self.device)

                optim.zero_grad()
                outputs, mean, log_var = self.forward(batch)
                loss = loss_function(batch, outputs, mean, log_var)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                train_acc += torch.sum(torch.pow((outputs-batch), 2))

            train_loss = train_loss/len(tr_loader.dataset)
            train_acc = math.sqrt(train_acc/len(tr_loader.dataset))  # reconstruction score

            self.eval()

            current_loss = 0.0
            current_acc = 0.0

            with torch.no_grad():
                for batch in tqdm(loader, desc=desc):

                    batch = batch.to(self.device)

                    outputs, mean, log_var = self.forward(batch)
                    loss = loss_function(batch, outputs, mean, log_var)

                    current_loss += loss.item()
                    current_acc += torch.sum(torch.pow((outputs-batch), 2))

                current_loss = current_loss/len(loader.dataset)
                current_acc = math.sqrt(current_acc/len(loader.dataset))  # reconstruction score

            print(output_print.format(epoch+1, num_epochs, train_loss, train_acc, current_loss, current_acc))


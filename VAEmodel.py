import math
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm
import wandb


def loss_function(x, x_prime, mean, log_var):
    kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return F.mse_loss(x_prime, x, reduction='sum') + kl_divergence


class ResNetVAE(nn.Module):
    def __init__(self, device, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256):
        super(ResNetVAE, self).__init__()

        self.device = device

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architectures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128                 # number of channels
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernel size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        resnet = models.resnet18(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten output of conv

        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.rand_like(std).to(self.device)  # normal standard random variable
            #std = logvar.mul(0.5).exp_()
            #eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, mu, logvar

    def train_model(self, tr_loader, optim, num_epochs, project_name):

        wandb.init(project=project_name, entity='niccolomalgeri')
        output_print = 'Epoch [{}/{}], train loss: {:.4f}, train reconstruction score: {:.4f}'

        for epoch in range(num_epochs):

            self.train()
            train_loss = 0.0
            train_acc = 0.0

            for batch, labels in tqdm(tr_loader, desc='training the model...'):
                batch = batch.to(self.device)

                optim.zero_grad()
                outputs, mean, log_var = self.forward(batch)
                loss = loss_function(batch, outputs, mean, log_var)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                mse = F.mse_loss(batch, outputs, reduction='sum')
                train_acc += mse.item()

            train_loss = train_loss / len(tr_loader.dataset)
            train_acc = math.sqrt(train_acc / len(tr_loader.dataset))  # reconstruction score

            wandb.log({'epoch': epoch + 1, 'training loss': train_loss, 'training reconstruction score': train_acc})

            print(output_print.format(epoch + 1, num_epochs, train_loss, train_acc))
            #torch.cuda.empty_cache()

        wandb.finish()

    def test_model(self, v_loader, t_loader, num_epochs, project_name, use_test=True):

        wandb.init(project=project_name, entity='niccolomalgeri')

        for epoch in range(num_epochs):

            if use_test:
                loader = t_loader
                desc = 'testing the model...'
                wandb_print_loss = 'test loss'
                wandb_print_acc = 'test reconstruction score'
            else:
                loader = v_loader
                desc = 'validating the model...'
                wandb_print_loss = 'val loss'
                wandb_print_acc = 'val reconstruction score'

            output_print = "Epoch [{}/{}], " + wandb_print_loss + ": {:.4f}, " + wandb_print_acc + ": {:.4f}"

            self.eval()
            current_loss = 0.0
            current_acc = 0.0

            with torch.no_grad():
                for batch, labels in tqdm(loader, desc=desc):
                    batch = batch.to(self.device)

                    outputs, mean, log_var = self.forward(batch)
                    loss = loss_function(batch, outputs, mean, log_var)

                    current_loss += loss.item()
                    current_acc += torch.sum(torch.pow((outputs - batch), 2))

                current_loss = current_loss / len(loader.dataset)
                current_acc = math.sqrt(current_acc / len(loader.dataset))  # reconstruction score

            wandb.log({'epoch': epoch + 1, wandb_print_loss: current_loss, wandb_print_acc: current_acc})

            print(output_print.format(epoch + 1, num_epochs, current_loss, current_acc))
            #torch.cuda.empty_cache()

        wandb.finish()
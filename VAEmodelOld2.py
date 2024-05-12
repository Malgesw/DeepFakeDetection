import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from tqdm.autonotebook import tqdm
import wandb
from typing import Tuple

# same VAE, but the decoder follows an alternative implementation


def loss_function(x, x_prime, mean, log_var):
    kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return kl_divergence + F.mse_loss(x, x_prime, reduction='sum')


class ReverseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batch1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.batch1(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.batch2(x)
        x = self.conv2(x)
        return x


class UnpoolingBlock(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.size)
        return x


class Resnet18Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.unpool1 = UnpoolingBlock(size=(2, 2))
        self.res1 = ReverseResidualBlock(512, 512, 3, 1, 1)
        self.res2 = ReverseResidualBlock(512, 256, 3, 2, 1)
        self.res3 = ReverseResidualBlock(256, 256, 3, 1, 1)
        self.res4 = ReverseResidualBlock(256, 128, 3, 2, 1)
        self.res5 = ReverseResidualBlock(128, 128, 3, 1, 1)
        self.res6 = ReverseResidualBlock(128, 64, 3, 2, 1)
        self.res7 = ReverseResidualBlock(64, 64, 3, 1, 1)
        self.res8 = ReverseResidualBlock(64, 64, 3, 1, 1)
        self.unpool2 = UnpoolingBlock(size=(112, 112))
        self.conv1 = nn.ConvTranspose2d(64, 3, 7, 2, 3)
        self.unpool3 = UnpoolingBlock(size=(224, 224))

    def forward(self, z):
        x = self.fc1(z)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.unpool1(x)
        x = nn.functional.relu(x)
        x = nn.functional.relu(self.res1(x))
        x = nn.functional.relu(self.res2(x))
        x = nn.functional.relu(self.res3(x))
        x = nn.functional.relu(self.res4(x))
        x = nn.functional.relu(self.res5(x))
        x = nn.functional.relu(self.res6(x))
        x = nn.functional.relu(self.res7(x))
        x = nn.functional.relu(self.res8(x))
        x = nn.functional.relu(self.unpool2(x))
        x = self.conv1(x)
        x = self.unpool3(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), 3, 224, 224)
        return x


class Resnet18VAE(nn.Module):

    def __init__(self, device, latent_dim=256, hidden_dim=1024):
        super(Resnet18VAE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(*list(models.resnet18(weights='IMAGENET1K_V1').children())[:-1])  # remove original fc layer
        self.hidden_fc1 = nn.Linear(512, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # batch normalization of a fc layer before passing x to mean and logvar layers

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = Resnet18Decoder(latent_dim=latent_dim)

    def reparametrize(self, mean, log_var):
        std = torch.exp(log_var/2)
        epsilon = torch.rand_like(std).to(self.device)  # normal standard random variable
        return std * epsilon + mean  # z = std_dev(x)*epsilon + mean(x)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
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

            train_loss = train_loss/len(tr_loader.dataset)
            train_acc = math.sqrt(train_acc/len(tr_loader.dataset))  # reconstruction score

            wandb.log({'epoch': epoch + 1, 'training loss': train_loss, 'training reconstruction score': train_acc})

            print(output_print.format(epoch+1, num_epochs, train_loss, train_acc))
            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()

        wandb.finish()
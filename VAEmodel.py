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
from typing import Tuple, List, Sequence
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE



# VAE structure based on hsynilin19's implementation (https://github.com/hsinyilin19/ResNetVAE/tree/master)


def normalize_tensor(t: torch.Tensor, per_channel=True) -> torch.Tensor:
    batch_size = t.shape[0]
    if per_channel:
        # each channel is normalized separately
        min_vals = t.view(batch_size, 3, -1).min(dim=2, keepdim=True).values
        max_vals = t.view(batch_size, 3, -1).max(dim=2, keepdim=True).values
        min_vals = min_vals.view(batch_size, 3, 1, 1)
        max_vals = max_vals.view(batch_size, 3, 1, 1)
    else:
        # single normalization for the three channels
        min_vals = t.view(batch_size, -1).min(dim=1, keepdim=True).values
        max_vals = t.view(batch_size, -1).max(dim=1, keepdim=True).values
        min_vals = min_vals.view(batch_size, 1, 1, 1)
        max_vals = max_vals.view(batch_size, 1, 1, 1)

    norm_tensor = (t - min_vals) / (max_vals - min_vals)
    return norm_tensor


def loss_function(x, x_prime, mean, log_var, use_mean=False):
    kl_divergence = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if use_mean:
        reduction = 'mean'
    else:
        reduction = 'sum'

    recon_loss = F.mse_loss(x_prime, x, reduction=reduction)
    return recon_loss + kl_divergence


class ResNetVAE(nn.Module):
    def __init__(self, device, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256):
        super(ResNetVAE, self).__init__()

        self.device = device

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architectures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128  # number of channels
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernel size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        resnet = models.resnet18(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
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
            #nn.BatchNorm2d(3, momentum=0.01),
            #nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

        # PRECISION RECALL CURVES SINGLE PLOT
        self.prec_recall_curves: List[Tuple[List[float], List[float], int]] = []
        self.prec_recall_auc: List[float] = []

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
        return x_prime, mu, logvar, z

    def train_model(self, tr_loader, optim, num_epochs, project_name='', wandb_log=False, use_mean=False):

        if wandb_log:
            wandb.init(project=project_name, entity='niccolomalgeri',
                       name='batch size: {}, use mean?: {}, training samples: {}'.format(
                           len(tr_loader.dataset) / len(tr_loader),
                           use_mean, len(tr_loader.dataset)))

        output_print = 'Epoch [{}/{}], train loss: {:.4f}, train reconstruction score: {:.4f}'

        for epoch in range(num_epochs):

            self.train()
            train_loss = 0.0
            train_acc = 0.0

            for batch, labels in tqdm(tr_loader, desc='training the model...'):
                batch = batch.to(self.device)

                optim.zero_grad()
                outputs, mean, log_var, _ = self.forward(batch)
                #norm_batch = normalize_tensor(batch, per_channel=True)
                # print(norm_batch)
                # print('\n')
                # print(outputs)
                #norm_batch = norm_batch.to(self.device)
                loss = loss_function(batch, outputs, mean, log_var, use_mean=use_mean)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                mse = F.mse_loss(batch, outputs, reduction='sum')
                train_acc += mse.item()

            train_loss = train_loss / len(tr_loader.dataset)
            train_acc = math.sqrt(train_acc / len(tr_loader.dataset))  # reconstruction score

            if wandb_log:
                wandb.log({'epoch': epoch + 1, 'training loss': train_loss, 'training reconstruction score': train_acc})

            print(output_print.format(epoch + 1, num_epochs, train_loss, train_acc))

        if wandb_log:
            wandb.finish()

    def test_model(self, v_loader, t_loader, num_epochs, project_name='', use_test=True, wandb_log=False, batch_size=50,
                   use_mean=False, test_fake_samples=10, current_epoch=0, saved_scores_path=''):

        if saved_scores_path != '':
            path = saved_scores_path
        else:
            path = './plots'

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

        if wandb_log:
            wandb.init(project=project_name, entity='niccolomalgeri',
                       name='test set?: {}, batch size: {}, use mean?: {}, test fake samples: {} over {}'.format(
                           use_test, batch_size, use_mean, test_fake_samples, int(len(loader.dataset))))

        reconstruction_scores = []
        labs = []
        tsne_scores = np.empty((0, self.CNN_embed_dim))
        tsne_labs = []

        for epoch in range(num_epochs):

            self.eval()
            current_loss = 0.0
            current_acc = 0.0

            with torch.no_grad():
                for batch, labels in tqdm(loader, desc=desc):

                    batch = batch.to(self.device)

                    outputs, mean, log_var, latent_outputs = self.forward(batch)
                    #norm_batch = normalize_tensor(batch, per_channel=True)
                    #print(norm_batch)
                    #print('\n')
                    #print(outputs)
                    #norm_batch = norm_batch.to(self.device)
                    loss = loss_function(batch, outputs, mean, log_var)

                    current_loss += loss.item()
                    mse = F.mse_loss(batch, outputs, reduction='none')
                    mse = mse.mean(dim=[1, 2, 3])
                    #current_acc += mse.item()

                    if epoch == num_epochs - 1:
                        tsne_labs += labels
                        tsne_scores = np.concatenate((tsne_scores, latent_outputs.cpu().numpy()), axis=0)
                        reconstruction_scores.extend(mse.cpu().tolist())
                        for label in labels:
                            labs.append(label)

                current_loss = current_loss / len(loader.dataset)
                #current_acc = math.sqrt(current_acc / len(loader.dataset))  # reconstruction score

            if wandb_log:
                wandb.log({'epoch': epoch + 1, wandb_print_loss: current_loss, wandb_print_acc: current_acc})

            print(output_print.format(epoch + 1, num_epochs, current_loss, current_acc))

        # PLOTS ---------------------------------------------------------------------------------------
        labs = list(map(int, labs))
        torch.save({
            'scores-labels': dict(zip(reconstruction_scores, labs))
        }, saved_scores_path + '/reconstruction_scores.pth')

        # HISTOGRAM -----------------------------------------------------------------------------------
        x_fakes = [score for score, label in zip(reconstruction_scores, labs) if label == 0]
        x_real = [score for score, label in zip(reconstruction_scores, labs) if label == 1]

        min_val = min(min(x_fakes), min(x_real))
        max_val = max(max(x_fakes), max(x_real))
        num_bins = 30
        bins = np.linspace(min_val, max_val,
                           num_bins)  # now every bin in the two histograms should be of the same width

        plt.figure(figsize=(8, 8))
        plt.hist(x_fakes, bins=bins, edgecolor='black', linewidth=1.2, density=True, alpha=0.5, label='Fake')
        plt.hist(x_real, bins=bins, edgecolor='black', linewidth=1.2, density=True, alpha=0.5, label='Real')

        plt.xlabel('Reconstruction Scores')
        plt.ylabel('Frequency')
        plt.title(
            'Histogram of Image Reconstruction Scores \n(use mean: {}, batch size: {}, test fake samples: {}/{}, current epoch: {})'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig(
            path + '/histogram(use mean: {}, batch size: {}, test fake samples: {} over {}, current epoch: {}).jpg'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.show()

        # PRECISION-RECALL CURVE ----------------------------------------------------------------------
        reversed_recon_scores = np.subtract(1, np.array(reconstruction_scores))
        reversed_recon_scores = reversed_recon_scores.tolist()

        precision, recall, thresholds = precision_recall_curve(labs, reversed_recon_scores)
        area_under_curve = average_precision_score(labs, reversed_recon_scores)
        self.prec_recall_auc.append(area_under_curve)
        self.prec_recall_curves.append((recall, precision, current_epoch))
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            'Precision-Recall Curve of Reconstruction Scores \n(use mean: {}, batch size: {}, test fake samples: {}/{}, current epoch: {})'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.savefig(
            path + '/precisionrecallcurve(use mean: {}, batch size: {}, test fake samples: {} over {}, current epoch: {}).png'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.show()

        # TSNE LATENT SPACE PLOT -----------------------------------------------------------------------------------
        tsne = TSNE(n_components=2).fit_transform(tsne_scores)
        tsne_labs = list(map(int, tsne_labs))
        tsne_labs = np.array(tsne_labs)
        plt.figure(figsize=(8, 8))
        colors = [0, 1]

        for color in colors:
            plt.scatter(tsne[tsne_labs == color, 0], tsne[tsne_labs == color, 1],
                        label=('Real' if color == 1 else 'Fake'))

        plt.legend(loc='upper right')
        plt.title(
            'TSNE plot of Reconstruction Scores \n(use mean: {}, batch size: {}, test fake samples: {}/{}, current epoch: {})'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.savefig(
            path + '/TSNE_latent_space(use mean: {}, batch size: {}, test fake samples: {} over {}, current epoch: {}).png'.format(
                use_mean, batch_size, test_fake_samples, int(len(loader.dataset)), current_epoch))
        plt.show()

        if wandb_log:
            wandb.finish()

    def train_test_save_model(self, tr_loader, t_loader, optim, num_epochs: Tuple[int, int], train_project_name='',
                              test_project_name='', wandb_log=False,
                              use_mean=False, batch_size=150, test_fake_samples=250, saved_models_path='',
                              saved_scores_path=''):

        if wandb_log:
            wandb.init(project=train_project_name, entity='niccolomalgeri',
                       name='batch size: {}, use mean?: {}, training samples: {}'.format(
                           len(tr_loader.dataset) / len(tr_loader),
                           use_mean, len(tr_loader.dataset)))

        output_print = 'Epoch [{}/{}], train loss: {:.4f}, train reconstruction score: {:.4f}'

        train_epochs, test_epochs = num_epochs

        for epoch in range(train_epochs):

            self.train()
            train_loss = 0.0
            train_acc = 0.0

            for batch, labels in tqdm(tr_loader, desc='training the model...'):
                batch = batch.to(self.device)

                optim.zero_grad()
                outputs, mean, log_var, _ = self.forward(batch)
                loss = loss_function(batch, outputs, mean, log_var, use_mean=use_mean)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                mse = F.mse_loss(batch, outputs, reduction='sum')
                train_acc += mse.item()

            train_loss = train_loss / len(tr_loader.dataset)
            train_acc = math.sqrt(train_acc / len(tr_loader.dataset))  # reconstruction score

            if wandb_log:
                wandb.log({'epoch': epoch + 1, 'training loss': train_loss, 'training reconstruction score': train_acc})

            print(output_print.format(epoch + 1, train_epochs, train_loss, train_acc))

            if ((epoch + 1) % 10) == 0:
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, saved_models_path + '/resnetVAE_' + str((epoch + 1) / 10) + '.pth')

                self.test_model(t_loader, t_loader, num_epochs=test_epochs, project_name=test_project_name,
                                use_test=True, wandb_log=False, batch_size=batch_size, use_mean=use_mean,
                                test_fake_samples=test_fake_samples,
                                current_epoch=epoch + 1, saved_scores_path=saved_scores_path)

        plt.figure(figsize=(8, 8))

        for recall, precision, epoch in self.prec_recall_curves:
            plt.plot(recall, precision, marker='.', label='Epoch {}'.format(epoch))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            'Precision-Recall Curve of Reconstruction Scores \n(use mean: {}, batch size: {}, test fake samples: {}/{}, epochs: {})'.format(
                use_mean, batch_size, test_fake_samples, int(len(t_loader.dataset)), test_epochs))
        plt.legend()
        plt.savefig(
            './saved_scores/precisionrecallcurve(use mean: {}, batch size: {}, test fake samples: {} over {}, epochs: {}).png'.format(
                use_mean, batch_size, test_fake_samples, int(len(t_loader.dataset)), test_epochs))
        plt.show()

        torch.save({
            'areas_under_curve': self.prec_recall_auc
        }, saved_scores_path + '/PRC_AUC.pth')

        if wandb_log:
            wandb.finish()

    def plots_test_model(self, t_loader, num_epochs, images_dim: Tuple[int, int, int],
                         use_mean=False, test_fake_samples=10):

        desc = 'testing the model...'
        wandb_print_loss = 'test loss'
        wandb_print_acc = 'test reconstruction score'

        output_print = "Epoch [{}/{}], " + wandb_print_loss + ": {:.4f}, " + wandb_print_acc + ": {:.4f}"

        reconstruction_scores = []
        labs = []
        image_errors = np.empty((0, *images_dim))
        original_images = np.empty((0, *images_dim))
        reconstructed_images = np.empty((0, *images_dim))

        for epoch in range(num_epochs):

            self.eval()
            current_loss = 0.0
            current_acc = 0.0

            with torch.no_grad():
                for batch, labels in tqdm(t_loader, desc=desc):

                    batch = batch.to(self.device)

                    outputs, mean, log_var, _ = self.forward(batch)
                    loss = loss_function(batch, outputs, mean, log_var, use_mean=use_mean)

                    current_loss += loss.item()

                    mse = F.mse_loss(batch, outputs, reduction='none')
                    mse = mse.mean(dim=[1, 2, 3])

                    if epoch == num_epochs - 1:
                        image_errors = np.concatenate((image_errors, torch.abs(torch.sub(batch, outputs)).cpu().numpy()), axis=0)
                        original_images = np.concatenate((original_images, batch.cpu().numpy()), axis=0)
                        reconstructed_images = np.concatenate((reconstructed_images, outputs.cpu().numpy()), axis=0)
                        reconstruction_scores.extend(mse.cpu().tolist())

                        for label in labels:
                            labs.append(label)

                current_loss = current_loss / len(t_loader.dataset)

            print(output_print.format(epoch + 1, num_epochs, current_loss, current_acc))

        # PLOT IMAGE ERRORS
        print(image_errors.shape)
        labs = list(map(int, labs))
        real_labs = [i for i, l in enumerate(labs) if l == 1]
        fakes_labs = [i for i, l in enumerate(labs) if l == 0]
        real_errors = image_errors[real_labs]
        fakes_errors = image_errors[fakes_labs]

        images = {'Original real': original_images[real_labs], 'Original fakes': original_images[fakes_labs],
                  'Reconstructed real': reconstructed_images[real_labs], 'Reconstructed fakes': reconstructed_images[fakes_labs]}

        _, axs_real = plt.subplots(2, real_errors.shape[0] // 2, figsize=(10, 5))

        plt.suptitle('Reconstruction errors of real test images (samples: {}/{}, test epochs: {})'.format(
            len(t_loader.dataset) - test_fake_samples, len(t_loader.dataset), num_epochs))

        for i, ax in enumerate(axs_real.flat):
            image = np.transpose(real_errors[i], (1, 2, 0))
            ax.imshow(image)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./saved_test_images/real_recon_scores(samples: {} over {}, test epochs: {}).png'.format(
            len(t_loader.dataset) - test_fake_samples, len(t_loader.dataset), num_epochs))
        plt.show()

        _, axs_fakes = plt.subplots(2, fakes_errors.shape[0] // 2, figsize=(10, 5))

        plt.suptitle('Reconstruction errors of fake test images (samples: {}/{}, test epochs: {})'.format(
            test_fake_samples, len(t_loader.dataset), num_epochs))

        for i, ax in enumerate(axs_fakes.flat):
            image = np.transpose(fakes_errors[i], (1, 2, 0))
            ax.imshow(image)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./saved_test_images/fakes_recon_scores(samples: {} over {}, test epochs: {}).png'.format(
            test_fake_samples, len(t_loader.dataset), num_epochs))
        plt.show()

        _, axs_real_freq = plt.subplots(2, real_errors.shape[0] // 2, figsize=(10, 5))

        plt.suptitle('Reconstruction errors frequencies of real test images (samples: {}/{}, test epochs: {})'.format(
            len(t_loader.dataset)-test_fake_samples, len(t_loader.dataset), num_epochs))

        for i, ax in enumerate(axs_real_freq.flat):

            fft = np.fft.fftn(np.transpose(real_errors[i], (1, 2, 0)), axes=(0, 1, 2))
            norm_fft = np.abs(fft)
            spectrum = np.fft.fftshift(norm_fft)
            spectrum = np.log(1 + spectrum)
            norm_spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
            ax.imshow(norm_spectrum, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./saved_test_images/real_recon_error_freqs.png')
        plt.show()

        _, axs_fakes_freq = plt.subplots(2, fakes_errors.shape[0] // 2, figsize=(10, 5))

        plt.suptitle('Reconstruction errors frequencies of fake test images (samples: {}/{}, test epochs: {})'.format(
            test_fake_samples, len(t_loader.dataset), num_epochs))

        for i, ax in enumerate(axs_fakes_freq.flat):

            fft = np.fft.fftn(np.transpose(fakes_errors[i], (1, 2, 0)), axes=(0, 1, 2))
            norm_fft = np.abs(fft)
            spectrum = np.fft.fftshift(norm_fft)
            spectrum = np.log(1 + spectrum)
            norm_spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
            ax.imshow(norm_spectrum, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./saved_test_images/fakes_recon_error_freqs.png')
        plt.show()

        for t, imgs in images.items():
            _, axs = plt.subplots(2, imgs.shape[0] // 2, figsize=(10, 5))

            plt.suptitle(t)

            for i, ax in enumerate(axs.flat):
                image = np.transpose(imgs[i], (1, 2, 0))
                ax.imshow(image)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig('./saved_test_images/' + str.lower(t) + '.png')
            plt.show()
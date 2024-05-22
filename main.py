import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple
from VAEmodel import ResNetVAE
import os
import shutil
import random


def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def extract_images(path_name, done=False):

    if not done:
        train_sub_dirs = [d for d in os.listdir(path_name) if os.path.isdir(os.path.join(path_name, d))]

        for train_sub_dir in train_sub_dirs:

            train_sub_dir_path = os.path.join(path_name, train_sub_dir)

            for image in os.listdir(train_sub_dir_path):
                base, ext = os.path.splitext(image)
                old_image_path = os.path.join(train_sub_dir_path, image)
                new_image_path = os.path.join(train_sub_dir_path, f"{base}_{train_sub_dir}{ext}")
                os.rename(old_image_path, new_image_path)

            for image in os.listdir(train_sub_dir_path):
                image_path = os.path.join(train_sub_dir_path, image)
                new_path = os.path.join(path_name, image)
                shutil.move(image_path, new_path)

            shutil.rmtree(train_sub_dir_path)


def organize_data_folder(transform, reals_path, fakes_path, main_data_path, train_samples=99850, test_fake_samples=100) -> Tuple[ImageFolder, ImageFolder]:

    extract_images(path_name=reals_path, done=False)
    extract_images(path_name=fakes_path, done=False)

    train_real_path = os.path.join(main_data_path, 'train/real')
    test_real_path = os.path.join(main_data_path, 'test/real')
    test_fake_path = os.path.join(main_data_path, 'test/fake')

    os.makedirs(train_real_path, exist_ok=True)
    os.makedirs(test_real_path, exist_ok=True)
    os.makedirs(test_fake_path, exist_ok=True)

    clean_folder(train_real_path)
    clean_folder(test_real_path)
    clean_folder(test_fake_path)

    real_images = os.listdir(reals_path)
    fake_images = os.listdir(fakes_path)

    print(len(real_images))
    print(len(fake_images))

    train_images = random.sample(real_images, train_samples)
    test_fake_images = random.sample(fake_images, test_fake_samples)  # 100 fake images
    test_real_images = list((set(real_images) - set(train_images)))  # 100 real images

    for image in train_images:
        old_path = str(os.path.join(reals_path, image))
        new_path = os.path.join(train_real_path, image)
        shutil.copy(old_path, new_path)

    for image in test_real_images:
        old_path = str(os.path.join(reals_path, image))
        new_path = os.path.join(test_real_path, image)
        shutil.copy(old_path, new_path)

    for image in test_fake_images:
        old_path = str(os.path.join(fakes_path, image))
        new_path = os.path.join(test_fake_path, image)
        shutil.copy(old_path, new_path)

    train_path = os.path.join(main_data_path, 'train')
    test_path = os.path.join(main_data_path, 'test')

    train_set = ImageFolder(train_path, transform=transform)
    test_set = ImageFolder(test_path, transform=transform)
    print(test_set.class_to_idx)

    return train_set, test_set


def train_test_model(t_f_samples, b_sizes, use_m, model: ResNetVAE, optim, train_epochs=15, test_epochs=10):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_path = './data/FF++/original_sequences'
    main_path = './data/FF++'
    fake_path = './data/FF++/Deepfakes'

    for tfs in t_f_samples:
        for bs in b_sizes:
            for um in use_m:

                train_set, test_set = organize_data_folder(transform, real_path, fake_path, main_path,
                                                           train_samples=89950, test_fake_samples=tfs)

                train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

                model.train_model(train_loader, optim, num_epochs=train_epochs, project_name='VAEresnet18_train_FF++',
                                  wandb_log=True, use_mean=um)

                model.test_model(test_loader, test_loader, num_epochs=test_epochs, project_name='VAEresnet18_test_FF++',
                                 use_test=True, wandb_log=True, batch_size=bs, use_mean=um, test_fake_samples=tfs)


def main():

    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'

    resnet18_vae = ResNetVAE(dev, 1024, 768, 256).to(dev)
    # pytorch_total_params = sum(p.numel() for p in resnet18_vae.parameters())
    # print(pytorch_total_params)
    optimizer = torch.optim.Adam(resnet18_vae.parameters(), lr=3e-4)

    test_fake_samples = [10, 100]
    batch_sizes = [50, 100, 150]
    use_mean = [True, False]

    train_test_model(test_fake_samples, batch_sizes, use_mean, resnet18_vae, optimizer, 15, 10)

    '''for batch, labels in train_loader:
        sample = batch
        break

    img = sample[0]
    output, m, v = resnet18_vae.forward(sample.to(dev))
    #tr = transforms.Resize(32)
    #img = tr(img)
    img = img.permute(1, 2, 0).detach().numpy()
    norm_img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(norm_img, interpolation='bicubic')
    plt.savefig('./plots/FF_input_image.jpg')
    plt.show()

    img2 = output[0]
    #img2 = tr(img2)
    img2 = img2.permute(1, 2, 0).cpu().detach().numpy()
    norm_img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    plt.imshow(norm_img2, interpolation='bicubic')
    plt.savefig('./plots/FF_reconstructed_image.jpg')
    plt.show()'''


if __name__ == "__main__":
    main()




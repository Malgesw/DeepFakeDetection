import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from VAEmodelOld import Resnet18VAE
from VAEmodel import Resnet18VAE

def main():

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = 'data/CIFAKE/train/one_class_classification'
    test_path = './data/CIFAKE/test'

    train_set = ImageFolder(train_path, transform=transform)

    train_size = int(0.1 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    test_set = ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'

    torch.cuda.empty_cache()

    #resnet18_vae = Resnet18VAE(dev, 256, 1024).to(dev)
    resnet18_vae = Resnet18VAE(dev, 10, 2048).to(dev)
    pytorch_total_params = sum(p.numel() for p in resnet18_vae.parameters())
    print(pytorch_total_params)
    optimizer = torch.optim.Adam(resnet18_vae.parameters(), lr=3e-4)
    resnet18_vae.train_model(train_loader, optimizer, num_epochs=20, project_name='VAEresnet18_train')
    #resnet18_vae.test_model(test_loader, test_loader, num_epochs=15, use_test=True)

    for batch, labels in train_loader:
        sample = batch
        break

    img = sample[0]
    tr = transforms.Resize(32)
    img = tr(img)
    img = img.permute(1, 2, 0).detach().numpy()
    norm_img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(norm_img, interpolation='bicubic')
    plt.show()

    output, m, v = resnet18_vae.forward(sample.to(dev))
    img2 = output[0]
    img2 = tr(img2)
    img2 = img2.permute(1, 2, 0).cpu().detach().numpy()
    norm_img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    plt.imshow(norm_img2, interpolation='bicubic')
    plt.show()


if __name__ == "__main__":
    main()




import torch.cuda
import model


def main():

    # TODO: import FF++ dataset
    train_set, test_set = ..., ...
    train_loader, test_loader = ..., ...
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'

    resnet18_vae = model.Resnet18VAE(dev, 256, 1024)
    optimizer = model.torch.optim.Adam(resnet18_vae.parameters(), lr=3e-4)
    resnet18_vae.train_model(train_loader, test_loader, test_loader, optimizer, num_epochs=20, use_test=True)


if __name__ == "__main__":
    main()




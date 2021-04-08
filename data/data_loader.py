import torch
from torchvision import datasets, transforms
from torch.utils import data
from data.dataset import *
from torchvision.datasets import SVHN, MNIST, FashionMNIST, EMNIST, ImageFolder

def get_loader(dataset, path, bsz, cifar_c_type = None):
    
    if dataset == 'cifar':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1944, 0.2010)

        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std) ])

        train_dataset = datasets.CIFAR10(root = path, 
                                         train = True, 
                                         download = True, 
                                         transform = train_transform)
        
        num_train  = len(train_dataset)
        indices    = torch.randperm(num_train).tolist()
        valid_size = 2048

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
        print(len(valid_idx))

        train_dataset = data.Subset(train_dataset, train_idx)
        valid_dataset = data.Subset(train_dataset, valid_idx)
        print("train_dataset length: ", len(train_dataset))
        print("valid_dataset length: ", len(valid_dataset))

        train_loader = data.DataLoader(train_dataset,
                                       batch_size = bsz,
                                       shuffle = True,
                                       drop_last = True)

        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size = 1024,
                                       shuffle = True,
                                       drop_last = True)

        return (train_loader, valid_loader)

    elif dataset == 'cifar_c': 
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1944, 0.2010)

        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std) ])
        
        cifar_c = CIFAR10_C(path,
                            c_type = cifar_c_type,
                            transform = train_transform)

        valid_c_loader = data.DataLoader(cifar_c,
                                         batch_size = 500,
                                         shuffle = False)

        return valid_c_loader
    elif dataset == 'mnist':
        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = 0.1307, std = 0.3081) ])
        mnist_dataset = MNIST(root = path, transform = train_transform)

        num_train = len(mnist_dataset)
        indices   = torch.randperm(num_train).tolist()
        valid_size = 2048

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_dataset = data.Subset(mnist_dataset, train_idx)
        valid_dataset = data.Subset(mnist_dataset, valid_idx)
        
        train_loader  = data.DataLoader(train_dataset,
                                        batch_size = bsz,
                                        shuffle = True,
                                        drop_last = True)

        valid_loader  = data.DataLoader(valid_dataset,
                                        batch_size = 2048,
                                        shuffle = True,
                                        drop_last = True)

        return (train_loader, valid_loader)
                                
    elif dataset == 'mnist_r':
        rot = [15, 30, 45, 60, 75]
        rot_loader = []
        for i in range(5):
            rotation = transforms.Compose([ transforms.RandomRotation(degrees = (rot[i], rot[i])),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = 0.1307, std = 0.3081)
                                          ])
            rotation_dataset = MNIST(root = path, transform = rotation)
            rot_loader.append(data.DataLoader(rotation_dataset, batch_size = 500, shuffle = False))
        
        return rot_loader
    elif dataset == 'svhn':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1944, 0.2010)

        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std) ])
        
        svhn_dataset = SVHN(path,
                            download = True,
                            transform = train_transform)

        svhn_loader  = data.DataLoader(dataset = svhn_dataset,
                                       batch_size = 500,
                                       shuffle = True)

        return svhn_loader

    elif dataset == 'lsun':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1944, 0.2010)

        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std) ])

        lsun_dataset = LSUN(path,
                            transform = train_transform)

        lsun_loader  = data.DataLoader(dataset = lsun_dataset,
                                       batch_size = 500,
                                       shuffle = True)

        return lsun_loader

    elif dataset == 'tiny':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1944, 0.2010)

        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std) ])

        tiny_dataset = TINY(path,
                            transform = train_transform)

        tiny_loader  = data.DataLoader(dataset = tiny_dataset,
                                       batch_size = 500,
                                       shuffle = True)

        return tiny_loader
    
    elif dataset == 'fmnist':
        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = 0.1307, std = 0.3081) ])

        fmnist_dataset  = FashionMNIST(root = '/home/jh/Documents/Research/Datasets', train = True,
                                       download = True, transform = train_transform)

        fmnist_loader   = data.DataLoader(dataset = fmnist_dataset, batch_size = 500)
        
        return fmnist_loader
    elif dataset == 'emnist':
        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = 0.1307, std = 0.3081) ])

        emnist_dataset  = EMNIST(root = '/home/jh/Documents/Research/Datasets', train = True,
                                 download = True, transform = train_transform)

        emnist_loader   = data.DataLoader(dataset = emnist_dataset, batch_size = 500)
        
        return emnist_loader

    elif dataset == 'nmnist':
        train_transform = transforms.Compose([ transforms.ToTensor(),
                                               transforms.Normalize(mean = 0.1307, std = 0.3081) ])

        nmnist_dataset  = ImageFolder(root = '/home/jh/Documents/Research/Datasets', train = True,
                                      download = True, transform = train_transform)

        nmnist_loader   = data.DataLoader(dataset = nmnist_dataset, batch_size = 500)

        return nmnist_loader



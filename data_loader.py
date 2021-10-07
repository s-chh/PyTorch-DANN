import torch
from torchvision import datasets
from torchvision import transforms
import os


def get_loader(args):
    if args.dset == 's2m':
        svhn_tr = transforms.Compose([transforms.Resize([32, 32]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])])
        s_train = datasets.SVHN(os.path.join(args.data_path, 'svhn'), split='train', download=True, transform=svhn_tr)
        s_test = datasets.SVHN(os.path.join(args.data_path, 'svhn'), split='test', download=True, transform=svhn_tr)

        mnist_tr = transforms.Compose([transforms.Resize([32, 32]),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])])
        t_train = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=True, download=True, transform=mnist_tr)
        t_test = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=False, download=True, transform=mnist_tr)

    elif args.dset == 'u2m':
        tr = transforms.Compose([transforms.Resize([32, 32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
        s_train = datasets.USPS(os.path.join(args.data_path, 'usps'), train=True, download=True, transform=tr)
        s_test = datasets.USPS(os.path.join(args.data_path, 'usps'), train=False, download=True, transform=tr)

        t_train = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=True, download=True, transform=tr)
        t_test = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=False, download=True, transform=tr)

    elif args.dset == 'm2u':
        tr = transforms.Compose([transforms.Resize([32, 32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
        s_train = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=True, download=True, transform=tr)
        s_test = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=False, download=True, transform=tr)

        t_train = datasets.USPS(os.path.join(args.data_path, 'usps'), train=True, download=True, transform=tr)
        t_test = datasets.USPS(os.path.join(args.data_path, 'usps'), train=False, download=True, transform=tr)

    elif args.dset == 'm2mm':
        mnist_tr = transforms.Compose([transforms.Resize([32, 32]),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])])

        s_train = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=True, download=True, transform=mnist_tr)
        s_test = datasets.MNIST(os.path.join(args.data_path, 'mnist'), train=False, download=True, transform=mnist_tr)

        mnistm_tr = transforms.Compose([transforms.Resize([32, 32]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])

        t_train = datasets.ImageFolder(root=os.path.join(args.data_path, 'mnistm', 'trainset'), transform=mnistm_tr)
        t_test = datasets.ImageFolder(root=os.path.join(args.data_path, 'mnistm', 'testset'), transform=mnistm_tr)

    elif args.dset == 'sd2sv':
        tr = transforms.Compose([transforms.Resize([32, 32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

        s_train = datasets.ImageFolder(root=os.path.join(args.data_path, 'sydigits', 'trainset'), transform=tr)
        s_test = datasets.ImageFolder(root=os.path.join(args.data_path, 'sydigits', 'trainset'), transform=tr)  # Does not have a testset

        t_train = datasets.SVHN(os.path.join(args.data_path, 'svhn'), split='train', download=True, transform=tr)
        t_test = datasets.SVHN(os.path.join(args.data_path, 'svhn'), split='test', download=True, transform=tr)

    elif args.dset == 'signs':
        tr = transforms.Compose([transforms.Resize([32, 32]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

        s_train = datasets.ImageFolder(root=os.path.join(args.data_path, 'sysigns', 'trainset'), transform=tr)
        s_test = datasets.ImageFolder(root=os.path.join(args.data_path, 'sysigns', 'trainset'), transform=tr)  # Does not have a testset

        t_train = datasets.ImageFolder(root=os.path.join(args.data_path, 'gtsrb', 'trainset'), transform=tr)
        t_test = datasets.ImageFolder(root=os.path.join(args.data_path, 'gtsrb', 'testset'), transform=tr)

    s_train_loader = torch.utils.data.DataLoader(dataset=s_train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True)

    s_test_loader = torch.utils.data.DataLoader(dataset=s_test,
                                                batch_size=args.batch_size * 2,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

    t_train_loader = torch.utils.data.DataLoader(dataset=t_train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True)

    t_test_loader = torch.utils.data.DataLoader(dataset=t_test,
                                                batch_size=args.batch_size * 2,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

    return s_train_loader, s_test_loader, t_train_loader, t_test_loader

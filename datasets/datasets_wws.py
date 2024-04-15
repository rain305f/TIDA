import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision import datasets, transforms
import torch
import pickle
import os
import math
from .randaugment import RandAugmentMC

# normalization parameters
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
tinyimagenet_mean, tinyimagenet_std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataset_class(args):
    if args.dataset == 'cifar10':
        return cifar10_dataset(args)
    elif args.dataset == 'cifar100':
        return cifar100_dataset(args)
    elif args.dataset == 'cifar100_20':
        return cifar100_20_dataset(args)
    elif args.dataset == 'tinyimagenet':
        return tinyimagenet_dataset(args)
    elif args.dataset in ['aircraft', 'stanfordcars', 'oxfordpets']:
        return generic224_dataset(args)
    elif args.dataset == 'imagenet100':
        return imagenet100_dataset(args)


def x_u_split_seen_novel(labels, lbl_percent,
              num_classes, lbl_set, unlbl_set, imb_factor):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        img_max = len(idx)
        num = img_max * ((1/imb_factor)**(i / (num_classes - 1.0)))
        idx = idx[:int(num)]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:])
        elif i in unlbl_set:
            unlabeled_idx.extend(idx)
    return labeled_idx, unlabeled_idx

#[4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 13, 5, 20, 8, 2] [4, 30, 55, 72, 25, 1, 32, 67, 73, 91, 13, 5, 20, 8, 2]

def x_u_split_seen_novel_with_sub(labels, lbl_percent,
            num_classes, lbl_set, unlbl_set, imb_factor, sub_index =[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        if i in sub_index:
            idx = np.where(labels == i)[0]
            np.random.shuffle(idx)
            img_max = len(idx)
            num = img_max * ((1/imb_factor)**(i / (num_classes - 1.0)))
            idx = idx[:int(num)]
            n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
            if i in lbl_set:
                labeled_idx.extend(idx[:n_lbl_sample])
                unlabeled_idx.extend(idx[n_lbl_sample:])
            elif i in unlbl_set:
                unlabeled_idx.extend(idx)
    return labeled_idx, unlabeled_idx


class cifar10_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR10SSL(self.data_root, train_labeled_idxs, train=True, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = CIFAR10SSL(self.data_root, train_unlabeled_idxs, train=True, transform=MyTransform(cifar10_mean,cifar10_std,self.transform_train), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        train_uncr_dataset = CIFAR10SSL_UNCR(self.data_root, train_unlabeled_idxs, train=True, transform=self.transform_train)
        test_dataset_seen = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(self.no_seen, self.no_class)))
        test_dataset_all = CIFAR10SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel

class cifar100_20_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel_with_sub(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)
        print(len(train_labeled_idxs))
        print(len(train_unlabeled_idxs))
        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR100SSL(self.data_root, train_labeled_idxs, train=True, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = CIFAR100SSL(self.data_root, train_unlabeled_idxs, train=True, transform=MyTransform(cifar100_mean,cifar100_std,self.transform_train), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset
        
        train_uncr_dataset = CIFAR100SSL_UNCR(self.data_root, train_unlabeled_idxs, train=True, transform=self.transform_train)
        # test_dataset_seen = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_seen = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49])
        test_dataset_novel = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=[ 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
        test_dataset_all = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False,labeled_set=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
 

#         2750 train_uncr_dataset
# 250 train_labeled_dataset
# 2750 train_unlabeled_dataset
#         print(len(train_uncr_dataset),"train_uncr_dataset")
#         print(len(train_labeled_dataset),"train_labeled_dataset")
#         print(len(train_unlabeled_dataset),"train_unlabeled_dataset")
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel



class cifar100_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR100SSL(self.data_root, train_labeled_idxs, train=True, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = CIFAR100SSL(self.data_root, train_unlabeled_idxs, train=True, transform=MyTransform(cifar100_mean,cifar100_std,self.transform_train), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        train_uncr_dataset = CIFAR100SSL_UNCR(self.data_root, train_unlabeled_idxs, train=True, transform=self.transform_train)
        test_dataset_seen = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False, labeled_set=list(range(self.no_seen, self.no_class)))
        test_dataset_all = CIFAR100SSL_TEST(self.data_root, train=False, transform=self.transform_val, download=False)
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel


class tinyimagenet_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomResizedCrop(64, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(tinyimagenet_mean, tinyimagenet_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)
        ])

        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformFixMatch(tinyimagenet_mean,tinyimagenet_std), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        train_uncr_dataset = GenericUNCR(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=self.transform_train)
        test_dataset_seen = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val, labeled_set=list(range(0,args.no_seen)))
        test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val, labeled_set=list(range(args.no_seen, args.no_class)))
        test_dataset_all = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val)
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel


class generic224_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
        ])

        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = GenericSSL(os.path.join(self.data_root, 'train'), train_labeled_idxs, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = GenericSSL(os.path.join(self.data_root, 'train'), train_unlabeled_idxs, transform=MyTransform(imgnet_mean,imgnet_std,self.transform_train), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        train_uncr_dataset = GenericUNCR(os.path.join(self.data_root, 'train'), train_unlabeled_idxs, transform=self.transform_train)
        test_dataset_seen = GenericTEST(os.path.join(self.data_root, 'test'), no_class=self.no_class, transform=self.transform_val, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = GenericTEST(os.path.join(self.data_root, 'test'), no_class=self.no_class, transform=self.transform_val, labeled_set=list(range(self.no_seen, self.no_class)))
        test_dataset_all = GenericTEST(os.path.join(self.data_root, 'test'), no_class=self.no_class, transform=self.transform_val)
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel



class imagenet100_dataset():
    def __init__(self, args):
        # augmentations
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, (0.2, 1.0)), #stronger augmnetation
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ])
# TransformFixMatch(cifar10_mean,cifar10_std)
        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
        ])

        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, transform=self.transform_train, temperature=self.temperature)
        train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformFixMatch(imgnet_mean,imgnet_std), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        train_uncr_dataset = GenericUNCR(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=self.transform_train)
        test_dataset_seen = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val, labeled_set=list(range(0,args.no_seen)))
        test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val, labeled_set=list(range(args.no_seen, args.no_class)))
        test_dataset_all = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=self.transform_val)
        return train_labeled_dataset, train_unlabeled_dataset, train_uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class MyTransform(object):
    def __init__(self, mean, std, transform):
        self.weak1 = transform
        self.weak2 = transform
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak1 = self.weak1(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        return weak1, weak2 , self.normalize(strong)

class MyTransform224(object):
    def __init__(self, mean, std, transform):
        self.weak1 = transform
        self.weak2 = transform
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak1 = self.weak1(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        return weak1, weak2 , self.normalize(strong)



class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, temperature=None, temp_uncr=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.temp[index]


class CIFAR10SSL_TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=True, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10SSL_UNCR(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.transform(img)
            img4 = self.transform(img)
            img5 = self.transform(img)
            img6 = self.transform(img)
            img7 = self.transform(img)
            img8 = self.transform(img)
            img9 = self.transform(img)
            img10 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, target, self.indexs[index]


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, temperature=None, temp_uncr=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = np.arange(len(self.targets))
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.temp[index]


class CIFAR100SSL_TEST(datasets.CIFAR100):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(100):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL_UNCR(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.transform(img)
            img4 = self.transform(img)
            img5 = self.transform(img)
            img6 = self.transform(img)
            img7 = self.transform(img)
            img8 = self.transform(img)
            img9 = self.transform(img)
            img10 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, target, self.indexs[index]


class GenericSSL(datasets.ImageFolder):
    def __init__(self, root, indexs, temperature=None, temp_uncr=None,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.temp[index]


class GenericTEST(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, labeled_set=None, no_class=200):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(no_class):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class GenericUNCR(datasets.ImageFolder):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.transform(img)
            img4 = self.transform(img)
            img5 = self.transform(img)
            img6 = self.transform(img)
            img7 = self.transform(img)
            img8 = self.transform(img)
            img9 = self.transform(img)
            img10 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, target, self.indexs[index]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)
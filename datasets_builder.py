import os
import PIL

from torchvision import datasets, transforms
from dataloader.Celeba import Celeba

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_transform(is_train,args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        degrees, translate, scale = args.random_affine
        transform = transforms.Compose([
        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),
        transforms.ColorJitter(args.color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
        ])
    
    # eval transform
    else:
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0

        size = int(args.input_size/crop_pct)    
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC)
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(t)
       
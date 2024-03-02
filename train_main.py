import argparse
import numpy as np
import os
import datetime
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

def get_args_parser():
    parser = argparse.ArgumentParser('VAE training', add_help=False)
    parser.add_argument('--is_train', default=True, type=int,
                        help='True for train, False for inference')
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=20, type=int)
    
    #Model parameters
    parser.add_argument('--input_size', default=28, type=int,
                        help='images input size')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes')
    parser.add_argument('--model', default='resnet_18', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_dims', default=1, type=int,
                        help='size of channels')
    
    # Encoder specifies
    parser.add_argument('--autoencoder_model', default='VanillaEncoder', type=str,
                        help='Name of model to train')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/Users/leo/Desktop/img_align_celeba', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Data agumentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                        help='Color jitter factor')
    parser.add_argument('--random_affine', nargs='*', default=[20, (0.1, 0.1), (0.9, 1.1)],
                    help='random affine arguments')
    
    return parser



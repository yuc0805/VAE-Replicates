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

from datasets_builder import build_dataset
import model_vae


def get_args_parser():
    parser = argparse.ArgumentParser('VAE training', add_help=False)
    parser.add_argument('--is_train', default=True, type=int,
                        help='True for train, False for inference')
    
    # training parameter
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=int,
                        help='Learning rate')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Numbers of epochs')
    
    #Model parameters
    parser.add_argument('--input_dims', default=64, type=int,
                        help='images input size (height or width)')
    parser.add_argument('--model', default='VanillaVAE', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--in_channels', default=3, type=int,
                        help='size of channels')
    parser.add_argument('--hidden_dims', default=None, type=list,
                        help='list of hidden dimension')
    parser.add_argument('--latent_dims', default=32, type=int,
                        help='size of latent dimension')
    
    
    
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



def main(args):
    device = torch.device(args.device)
    seed =  args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load dataset
    dataset = build_dataset(is_train=args.is_train,args=args)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=True)

    
    # initialize model
    model = model_vae.VAE(in_channels = args.in_channels,
                          latent_dims = args.latent_dims,
                          hidden_dims = args.hidden_dims,*args)
    model.to(device)

    if args.is_train:
        # training
        print("Training: Model = %s, Batch Size = %d, lr = %.2e" % (str(model), args.batch_size, args.lr))
        optimizer = torch.optim.Adam(model.parameters(),args.lr)
        #criterion = model_vae.VAE.forward_loss()

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

        for epoch in range(args.epochs):
            model.train()
            overall_loss = 0 
            for batch_idx, (x,_) in enumerate(dataloader):
                x = x.view(args.batch_size, args.input_dims)
                x = x.to(device)

                optimizer.zero_grad()

                x_hat, mu, logvar = model(x)
                loss = model.forward_loss(x,x_hat,logvar,mu)
                overall_loss += loss.item()

                loss.backward()
                optimizer.steop()

                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader), loss.item()))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # save model
        if args.output_dir:
            model_path = os.path.join(args.output_dir, 'VAE_model.pth')
            torch.save(model.state_dict(),model_path)

                
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    main(args)



             

        

    
    
        
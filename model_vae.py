import torch
import torch.nn as  nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,input_dims: int,
                 latent_dims: int,
                 hidden_dims: list=None,
                 *args):
        super(VAE,self).__init__()

        self.latent_dims = latent_dims
        self.device = args.device

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        
        if args.autoencoder_model == 'VanillaEncoder':
            # --------------------------------------------------------------------------
            # encoder specifics
            modules = []

            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=input_dims,out_channels=h_dim,
                                kernel_size=3,stride=2,padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU
                    )
                )
                input_dims = h_dim

            self.encoder = nn.Sequential(*modules)

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
            self.fc_var = nn.Linear(hidden_dims[-1]*4,latent_dims)

            # --------------------------------------------------------------------------
            
            # --------------------------------------------------------------------------
            # decoder specifics
            self.decoder_input = nn.Linear(latent_dims,hidden_dims[-1]*4)

            modules = []
            hidden_dims.reverse()

            for i in range(len(hidden_dims)-1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],hidden_dims[i+1],
                                           kernel_size=3,stride=2,padding=1,output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i+1]),
                        nn.LeakyReLU
                    )
                )
            
            self.decoder = nn.Sequential(*modules)
            # --------------------------------------------------------------------------
                
            # --------------------------------------------------------------------------
            # output layers
            self.output_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],hidden_dims[-1],
                                   kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1],out_channels=3,kernel_size=3,padding=1),
                nn.Tanh()
            )
            # --------------------------------------------------------------------------


    def forward_encoder(self,x):
        z = self.encoder(x)
        z = torch.flatten(z,start_dim=1)

        mu = self.fc_mu(z)
        logvar = self.fc_var(z)

        return mu,logvar
    
    def reparameterize(self, mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)

        return eps*std + mu
        
    def forward_decoder(self,z):
        x = self.decoder_input(z)
        x = x.view(-1,512,2,2)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x
    
    def forward_loss(self,imgs,pred,logvar,mu):
        recons_loss = F.mse_loss(pred,imgs)
        kld_loss = torch.mean(-0.5*torch.sum(1+logvar-mu**2-logvar.exp(),dim=1),dim=0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def forward(self,x):
        mu, logvar = self.forward_encoder(x)
        z = self.reparameterize(mu,logvar)
        x = self.forward_decoder(z)

    def sample(self,num_sample,device):
        z = torch.radn(num_sample,self.latent_dims)
        z = z.to(device)

    def generate(self,x,latent_dims,random_generate=False):
        if random_generate:
            eps = torch.rand_like(latent_dims)
            return self.forward_decoder(eps)[0]
        
        else:
            return self.forward(x)[0]
        




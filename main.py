from Project.GAN import Generator, Discriminator, weights_init
from Project.train import Train
import torch
import torch.nn as nn
ngpu = 0 #number of gpu available, use 0 for cpu

device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

#initialize Generator
gen = Generator(ngpu)
# Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(gen, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
gen.apply(weights_init)
# Print the model
print(gen)

#initialize Discriminator
disc = Discriminator(ngpu)
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
disc.apply(weights_init)
# Print the model
print(disc)

#parameters
lr = 2e-5
betas = (0.5, 0.5)
epoch = 5000
batch_size = 128
cuda = torch.cuda.is_available()
# Optimizers
optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)
trainer = Train(gen, disc, batch_size,cuda=cuda)
trainer.train_model(epochs=epoch)
from train_model import *
from matplotlib import pyplot as plt
import pandas as pd 
import torch
import torch.nn as nn

stocks = ["TMUS", "DIS", #Communication Services
          "AMZN", "TSLA", #Consumer Discretionary
          "KO", "WMT", #Consumer Staples
          "XOM", "CVX", #Energy
          "JPM", "BAC", #financial
          "JNJ", "PFE", #healthcare
          "UPS", "BA", #Industrials
          "AAPL", "MSFT", #Information Technology
          "LIN", "APD", #Materials
          "AMT", "PLD", #Real Estate
          "NEE", "DUK" #Utilities
          ]

num_simulations = 10

num_stocks = len(stocks)
b = 40
f = 20
k_s = 3

#parameters
lr = 1e-3
betas = (0.5, 0.5)
epoch = 50
batch_size = 128
gp_weight = 5

lr_exp_decay = 0.95

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = Generator(num_stocks, b, f, k_s)
disc = Discriminator(num_stocks, k_s)

cuda = torch.cuda.is_available()
# Optimizers
optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, lr_exp_decay)
lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, lr_exp_decay)
train_file = "test.npy"
trainer = Train(gen,disc, batch_size, optimizer_G,optimizer_D,lr_scheduler_G,lr_scheduler_D, train_file, cuda=cuda, gp_weight=gp_weight, test=True)

checkpoint = torch.load("./models/last_model.pt")
trainer.netG = checkpoint['model_G']
trainer.netD = checkpoint['model_D']

num_fake_batch = num_simulations
for i, sample_batched in enumerate(trainer.train_loader):
    x, y, A, scales= sample_batched['x'].to(trainer.device), sample_batched['y'].to(trainer.device), sample_batched['A'].to(trainer.device), sample_batched['scales'].to(trainer.device)
    fake_batches = []
    for z in range(num_fake_batch):
        Mf = trainer.gen_fake(x, A)
        print("Mf.shape", Mf.shape)
        fake_batches.append(torch.cat((x, Mf), dim=2).to(trainer.device))
    real_batch = torch.cat((x, y), dim=2).to(trainer.device)

    if i==0:
      break

np.store("porfolio_data/fake_batches.npy", fake_batches.numpy())
np.store("porfolio_data/real_batches.npy", real_batch.numpy())



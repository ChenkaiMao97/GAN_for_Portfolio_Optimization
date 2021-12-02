from train_model import *
from matplotlib import pyplot as plt
import pandas as pd 
import torch
import torch.nn as nn


stocks = ["AAPL", "TSLA", "GOOG", "IBM", "FB", "ZM", "AMC", "AMD", "PFE", "GM", "NVDA", "CVX", "CELG", "XOM", "AMZN"]

num_stocks = 15
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
train_file = "data1.npy"
trainer = Train(gen,disc, batch_size, optimizer_G,optimizer_D,lr_scheduler_G,lr_scheduler_D, train_file, cuda=cuda, gp_weight=gp_weight)
    

# train_result = df.to_numpy()
df = pd.read_csv('./'+'df.csv')
plt.figure(figsize=(7,7))
plt.plot(df['epoch'], df['train_G_loss'])
plt.plot(df['epoch'], df['train_D_loss'])
plt.plot(df['epoch'], df['test_G_loss'])
plt.plot(df['epoch'], df['test_D_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")

plt.legend(['train_G_loss', 'train_D_loss', 'test_G_loss', 'test_D_loss'])
plt.savefig("learning.png")


trainer = Train(gen,disc, batch_size, optimizer_G,optimizer_D,lr_scheduler_G,lr_scheduler_D, train_file, cuda=cuda, gp_weight=gp_weight)
checkpoint = torch.load("./models/last_model.pt")
trainer.netG = checkpoint['model_G']
trainer.netD = checkpoint['model_D']

num_fake_batch = 10
for i, sample_batched in enumerate(trainer.train_loader):
    x, y, A= sample_batched['x'].to(trainer.device), sample_batched['y'].to(trainer.device), sample_batched['A'].to(trainer.device)
    fake_batches = []
    for z in range(num_fake_batch):
        Mf = trainer.gen_fake(x, A)
        fake_batches.append(torch.cat((x, Mf), dim=2).to(trainer.device))
    real_batch = torch.cat((x, y), dim=2).to(trainer.device)

    if i==0:
      break

num_days = 2
num_stocks = 4
fig, axs = plt.subplots(num_days,num_stocks, figsize=(10,5))

for i in range(num_days):
	for j in range(num_stocks):
		axs[i,j].plot(list(range(real_batch.shape[-1])), real_batch.cpu()[i,j,:], linewidth=2)
		for k in range(num_fake_batch):
			axs[i,j].plot(list(range(real_batch.shape[-1])), fake_batches[k].detach().cpu()[i,j,:], linewidth=.5)
		axs[i,j].set_xlabel("date")
		axs[i,j].set_ylabel("Normalized Price")
		axs[i,j].set_title(stocks[j])

plt.tight_layout()
fig.savefig("price_prediction.png", dpi=1000)

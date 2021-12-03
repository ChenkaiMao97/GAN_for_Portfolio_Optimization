import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
# from torch.utils.data import Dataset, DataLoader

class PriceDataset(Dataset):
    def __init__(self, filename, b=40, f=20, total_sample_number=None, test=False):

        self.prices = np.load('./data/'+filename, allow_pickle=True).astype(np.float32)
        self.test = test

        if total_sample_number:
            random.seed(1234)
            indices = random.sample(list(range(self.prices.shape[0])), total_sample_number)
            self.prices = self.prices[indices, :, :]

        self.x = self.prices[:, :, :b]
        self.y = self.prices[:, :, b:b+f]
        if not self.test:
            self.A = self.prices[:, :, -1:]
        else:
            self.A = self.prices[:, :, -3:-2]
            self.scales = self.prices[:, :, -2:]

    def __len__(self):
        return self.prices.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx, :, :]
        y = self.y[idx, :, :]
        A = self.A[idx, :, :]

        if self.test:
            scales = self.scales[idx, :, :]
            sample = {'x': x, 'y': y, 'A': A, 'scales': scales}
        else:
            sample = {'x': x, 'y': y, 'A': A}

        return sample

class Generator(nn.Module):
    def __init__(self, in_c, b, f, kernel, pad=1, stride=2):
        super(Generator, self).__init__()
        # _W = int((b - kernel_size + 2 * padding) / stride + 1)
        self.c_conv1 = nn.Sequential(
            nn.Conv1d(in_c,2*in_c, kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        self.c_conv2 = nn.Sequential(
            nn.Conv1d(2*in_c,2*in_c, kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        self.c_conv3 = nn.Sequential(
            nn.Conv1d(2*in_c,2*in_c, kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        self.c_conv4 = nn.Sequential(
            nn.Conv1d(2*in_c,2*in_c, kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        self.c_fc = nn.Linear(6*in_c, in_c)
        
        # self.condition = nn.Sequential(
        #     self.conv1,
        #     self.conv,
        #     self.conv,
        #     self.conv,
        #     nn.Linear(, in_c) #input dim? activation?
        # )

        #Similator
        self.s_fc = nn.Linear(4*in_c, f*in_c)
        self.s_ct1 = nn.Sequential(
            nn.ConvTranspose1d(4*in_c, 2*in_c, kernel, stride=2, padding=pad, output_padding=pad),
            nn.ReLU(True)
        )

        self.s_ct2 = nn.Sequential(
            nn.ConvTranspose1d(2*in_c, in_c, kernel, stride=2, padding=pad, output_padding=pad)
        )

        # self.simulator = nn.Sequential(
        #     nn.Linear(4*in_c, f*in_c),
        #     nn.ConvTranspose1d(4*in_c, 2*in_c, kernel, stride=2),
        #     nn.ReLU(True),
        #     nn.ConvTranspose1d(2*in_c, in_c, kernel, stride=2),
        #     nn.ReLU(True)
        # )

    def forward(self, x, A, lam):
        B, k, b = x.shape
        # Conditioner
        # print("shapes: ", x.shape, A.shape, lam.shape)
        x = self.c_conv1(x)
        # print("shapes1: ", x.shape)
        x = self.c_conv2(x)
        # print("shapes2: ", x.shape)
        x = self.c_conv3(x)
        # print("shapes3: ", x.shape)
        # x = self.c_conv4(x).squeeze()
        x = self.c_conv4(x)
        # print("shapes4: ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.c_fc(x)
        # print("shapes5: ", x.shape)
        
        # print("shapes: ", x.shape, A.squeeze().shape, lam.shape)
        x = torch.cat((x, A.squeeze(), lam), dim=1).float()
        # print("shapes6: ", x.shape)

        # Simulator: 
        x = self.s_fc(x).reshape(B, 4*k, -1)
        # print("dense from simulator: ", x.shape)
        x = self.s_ct1(x)
        # print("shapes8: ", x.shape)
        x = self.s_ct2(x)
        # print("shapes9: ", x.shape)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_c, kernel):
        super(Discriminator, self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv1d(in_c, 2*in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(2*in_c, 4 * in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(4*in_c, 8 * in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(8*in_c, 16*in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(16*in_c, 32 * in_c, kernel, stride=2),
        #     nn.LeakyReLU(inplace=True),
        # )
        self.dense = nn.Linear(32 * in_c, 1) #input dim?

    def forward(self, x):
        # print("D initial: ", x.shape)
        x = self.conv1(x)
        # print("D cv1:", x.shape)
        x = self.conv2(x)
        # print("D cv2: ", x.shape)
        x = self.conv3(x)
        # print("D cv3: ", x.shape)
        x = self.conv4(x)
        # print("D cv4: ", x.shape)
        # x = self.conv5(x)
        # print("D cv5: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("Flattened: ", x.shape)
        x = self.dense(x)
        # print("dense shape", x.shape)
        return x

def weights_init(m):
    nn.init.normal_(m.weight.data, 0.0, 0.02)

# import torch
# import torch.nn as nn
# from torch.utils.data import random_split, DataLoader

class Train():
    def __init__(self, generator, discriminator, batch_size, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, train_file, cuda, gp_weight=10, critic_iter=5, test=False):
        self.netG = generator
        self.netD = discriminator
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.losses = {'G': [],'D': [], 'GP': []}
        self.train_file = train_file

        ds = PriceDataset(train_file, test=test)
        torch.manual_seed(42)
        train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])
        
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def gen_fake(self, x, A, seed=None):
        B, k, b = x.shape
        if seed:
            np.random.seed(seed)
        lam = torch.tensor(np.random.normal(0,1,(B,2*k))).to(self.device)
        return self.netG(x, A, lam)

    def train_model(self, epochs):
        df = pd.DataFrame(columns=['epoch','train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])

        for epoch in range(epochs):
            for i, sample_batched in enumerate(self.train_loader):
            
                x, y, A= sample_batched['x'].to(self.device), sample_batched['y'].to(self.device), sample_batched['A'].to(self.device)
        
                Mf = self.gen_fake(x, A)

                
                real_batch = torch.cat((x, y), dim=2).to(self.device)
                fake_batch = torch.cat((x, Mf), dim=2).to(self.device)
                # print("real_batch shape: ", real_batch.shape)
                # print("fake_batch shape: ", fake_batch.shape)
                

                real_D = self.netD(real_batch)
                fake_D = self.netD(fake_batch)

                # real_D = self.netD(torch.permute(real_batch, (0,2,1)))
                # fake_D = self.netD(torch.permute(fake_batch, (0,2,1)))

                grad_penalty = self.compute_gradient(real_batch, fake_batch)
                self.optimizer_D.zero_grad()
                d_loss = torch.mean(fake_D) - torch.mean(real_D) + grad_penalty
                d_loss.backward()
                self.optimizer_D.step()

                if i == self.critic_iter - 1:
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(float(grad_penalty.item()))
                    # print(d_loss, grad_penalty.item())
                
                #train generator every critic_iter steps 
                if i % self.critic_iter == 0:
                    self.optimizer_G.zero_grad()
                    Mf = self.gen_fake(x, A)
                    fake_batch_G = torch.cat((x, Mf), dim=2).to(self.device)
                    fake_critic = self.netD(fake_batch_G)
                    g_loss = -torch.mean(fake_critic)
                    g_loss.backward()
                    optimizer_G.step()

                    # print("Epoch: {}/{} \nD_loss: {:f} G_loss: {:f}".format(
                    #     epoch, epochs, d_loss.item(), g_loss.item()
                    # ))

            train_G_loss = 0
            train_D_loss = 0
            for i, sample_batched in enumerate(self.train_loader):
                x, y, A= sample_batched['x'].to(self.device), sample_batched['y'].to(self.device), sample_batched['A'].to(self.device)
                Mf = self.gen_fake(x, A)

                real_batch = torch.cat((x, y), dim=2).to(self.device)
                fake_batch = torch.cat((x, Mf), dim=2).to(self.device)

                real_D = self.netD(real_batch)
                fake_D = self.netD(fake_batch)
                
                train_G_loss += -torch.mean(fake_D)

                grad_penalty = self.compute_gradient(real_batch, fake_batch)
                train_D_loss += torch.mean(fake_D) - torch.mean(real_D) + grad_penalty
            
            train_G_loss /= self.train_loader.__len__()
            train_D_loss /= self.train_loader.__len__()
                
            
            test_G_loss = 0
            test_D_loss = 0
            for i, sample_batched in enumerate(self.test_loader):
                x, y, A= sample_batched['x'].to(self.device), sample_batched['y'].to(self.device), sample_batched['A'].to(self.device)
                Mf = self.gen_fake(x, A)

                real_batch = torch.cat((x, y), dim=2).to(self.device)
                fake_batch = torch.cat((x, Mf), dim=2).to(self.device)

                real_D = self.netD(real_batch)
                fake_D = self.netD(fake_batch)
                
                test_G_loss += -torch.mean(fake_D)

                grad_penalty = self.compute_gradient(real_batch, fake_batch)
                test_D_loss += torch.mean(fake_D) - torch.mean(real_D) + grad_penalty
            
            test_G_loss /= self.test_loader.__len__()
            test_D_loss /= self.test_loader.__len__()
            
            df = df.append({'epoch': epoch+1,
                            'train_G_loss': train_G_loss.item(),
                            'train_D_loss': train_D_loss.item(),
                            'test_G_loss': test_G_loss.item(),
                            'test_D_loss': test_D_loss.item(),
                          }, ignore_index=True)
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            print(f"epoch: {epoch}, train_G_loss: {train_G_loss}, train_D_loss: {train_D_loss}, test_G_loss: {test_G_loss}, test_D_loss, {test_D_loss}")
        
            checkpoint = {
                        'epoch': epoch,
                        'model_G': self.netG,
                        'model_D': self.netD,
                        'optimizer_G': self.optimizer_G,
                        'optimizer_D': self.optimizer_D,
                        'lr_scheduler_G': self.lr_scheduler_G,
                        'lr_scheduler_D': self.lr_scheduler_D
                     }
            torch.save(checkpoint, "./models/last_model.pt")

        return df

    def compute_gradient(self, real_data, fake_data):
        """Calculates the gradient penalty loss for WGAN GP"""
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1).to(self.device)
        eps = eps.expand_as(real_data).to(self.device)

        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data

        
        interpolation = interpolation.to(self.device)

        # get logits for interpolated images
        interp_logits = self.netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits).to(self.device)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        # add epsilon for stability
        # noise = 1e-10
        # grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, dtype=torch.double) + noise)
        return self.gp_weight * torch.mean((grad_norm - 1) ** 2)
        # return self.gp_weight * (torch.max(torch.zeros(1,dtype=torch.double).cuda() if self.use_cuda else torch.zeros(1,dtype=torch.double), gradients_norm.mean() - 1) ** 2), gradients_norm.mean().item()


if __name__ == '__main__':
    num_stocks = 22
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

    import torch
    import torch.nn as nn
    ngpu = 0 #number of gpu available, use 0 for cpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #initialize Generator
    gen = Generator(num_stocks, b, f, k_s)
    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netG = nn.DataParallel(gen, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    # gen.apply(weights_init)
    # Print the model
    # print(gen)

    #initialize Discriminator
    disc = Discriminator(num_stocks, k_s)
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    # disc.apply(weights_init)
    # Print the model
    # print(disc)

    cuda = torch.cuda.is_available()
    # Optimizers
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, lr_exp_decay)
    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, lr_exp_decay)
    train_file = "train.npy"
    trainer = Train(gen,disc, batch_size, optimizer_G,optimizer_D,lr_scheduler_G,lr_scheduler_D, train_file, cuda=cuda, gp_weight=gp_weight)
    df = trainer.train_model(epochs=epoch)

    df.to_csv('./'+'df.csv',index=False)




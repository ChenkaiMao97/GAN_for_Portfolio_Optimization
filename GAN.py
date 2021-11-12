import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, in_c, f, kernel):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # _W = int((W - kernel_size + 2 * padding) / stride + 1)
        self.conv = nn.Sequential(
            nn.Conv1d(2*in_c,2*in_c, kernel, stride=2),
            nn.ReLU()
        )
        self.condition = nn.Sequential(
            self.conv,
            self.conv,
            self.conv,
            self.conv,
            nn.Linear(2*in_c, in_c) #input dim? activation?
        )
        self.simulator = nn.Sequential(
            nn.Linear(in_c, f*in_c),
            nn.ConvTranspose1d(4*in_c, 2*in_c, kernel, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(2*in_c, in_c, kernel, stride=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.condition(x)
        x = self.simulator(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu, in_c, kernel):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, 2*in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(2*in_c, 4 * in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(4*in_c, 8 * in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(8*in_c, 16*in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(16*in_c, 32 * in_c, kernel, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.dense = nn.Linear(32 * in_c, 1) #input dim?

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return self.conv(x)

def weights_init(m):
    nn.init.normal_(m.weight.data, 0.0, 0.02)



import torch
import torch.nn as nn
class Train():
    def __init__(self, generator, discriminator, batch_size, cuda, gp_weight=10, critic_iter=5):
        self.netG = generator
        self.netD = discriminator
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        self.cuda = cuda
        if self.cuda:
            self.netG.cuda()
            self.netD.cuda()

    def train_model(self, epochs):
        for epoch in range(epochs):
            for i in range(self.critic_iter):

    def compute_gradient(self, real_data, fake_data):
        """Calculates the gradient penalty loss for WGAN GP"""
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)

        if self.cuda:
            eps = eps.cuda()

        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data

        if self.cuda:
            interpolation = interpolation.cuda()

        # get logits for interpolated images
        interp_logits = self.netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits).cuda() if self.cuda else \
            torch.ones_like(interp_logits)

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



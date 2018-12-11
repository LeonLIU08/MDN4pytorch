import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.FloatTensor')


class multi_bivariate_gaussian(nn.Module):
    def __init__(self, fea_w=60, fea_h=60, scale=1.0, nb_gau=1):
        super(multi_bivariate_gaussian, self).__init__()
        self.scale = scale
        self.nb_gau = nb_gau
        self.fea_h = fea_h
        self.fea_w = fea_w

        X = np.linspace(0, 1, fea_w)
        Y = np.linspace(0, 1, fea_h)
        X, Y = np.meshgrid(X, Y)
        self.pos = np.empty(X.shape + (2,))
        self.pos[:, :, 0] = X
        self.pos[:, :, 1] = Y
        self.pos = torch.from_numpy(self.pos)
        print('pos', self.pos.shape)
        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, inputs):
        pos = self.pos
        fea_map = torch.zeros((self.fea_h, self.fea_w), dtype=torch.float64)

        for i in range(self.nb_gau):
            fea_map = fea_map + self.scale * self.multivariate_gaussian(pos, inputs[i*2], inputs[i*2+1])
        # return self.global_pooling(fea_map.unsqueeze(0))

        return fea_map

    def multivariate_gaussian(self, pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        pi = 3.14159265359
        n = mu.shape[0]
        assert n == 2

        Sigma_det = torch.det(Sigma)
        Sigma_inv = torch.inverse(Sigma)
        N = torch.sqrt((2 * pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = torch.einsum('...k,kl,...l->...', [pos - mu, Sigma_inv, pos - mu])

        return torch.exp(-fac / 2) / N


mu1, mu2 = 0.5, 0.5  # (0, 1) tanh
mu3, mu4 = 0.3, 0.8
plt.figure(figsize=(12, 14))
for i in range(16):
    sig1, sig2 = 0.1, -0.2  # (0, 1) tanh
    ro = -0.95+i*0.125  # (0, 1) tanh
    net = multi_bivariate_gaussian(nb_gau=2, fea_h=60, fea_w=80)

    mu = np.array([mu1, mu2])
    Sigma = np.array([[sig1**2,       ro*sig1*sig2],
                      [ro*sig1*sig2,  sig2**2]])
    x1 = Variable(torch.from_numpy(mu), requires_grad=True)
    x2 = Variable(torch.from_numpy(Sigma), requires_grad=True)

    mu0 = np.array([mu3, mu4])
    Sigma1 = np.array([[sig1 ** 2, ro * sig1 * sig2],
                      [ro * sig1 * sig2, sig2 ** 2]])
    x3 = Variable(torch.from_numpy(mu0), requires_grad=True)
    x4 = Variable(torch.from_numpy(Sigma1), requires_grad=True)

    out = net([x1, x2, x3, x4])
    # print(out.grad_fn, out.shape)
    # y = torch.randn(1, 1, 1)
    # y = y.double()
    # out.backward(y)

    Z = out.data.numpy()

    plt.subplot(4, 4, i+1)
    plt.xlabel('sig1:%.3f, sig2:%.3f, ro:%.3f' % (sig1, sig2, ro))
    plt.imshow(Z)

plt.tight_layout()
plt.show()

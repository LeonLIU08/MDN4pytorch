import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class bivariate_gaussian(nn.Module):
    def __init__(self, fea_size=60, scale=10):
        super(bivariate_gaussian, self).__init__()
        self.scale = scale

        X = np.linspace(0, 60, fea_size)
        Y = np.linspace(0, 60, fea_size)
        X, Y = np.meshgrid(X, Y)
        self.pos = np.empty(X.shape + (2,))
        self.pos[:, :, 0] = X
        self.pos[:, :, 1] = Y
        self.pos = torch.from_numpy(self.pos)

        self.global_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        pos = self.pos
        fea_map = self.scale*self.multivariate_gaussian(pos, input[0], input[1])
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


net = bivariate_gaussian()
mu = np.array([30., 31.])
Sigma = np.array([[20., 1.5], [10, 45]])
x1 = Variable(torch.from_numpy(mu), requires_grad=True)
x2 = Variable(torch.from_numpy(Sigma), requires_grad=True)
out = net([x1, x2])
print(out.grad_fn, out.shape)
y = torch.randn(1, 1, 1)
y = y.double()
# out.backward(y)


Z = out.data.numpy()
# Create a surface plot and projected filled contour plot under it.
N = 60
X = np.linspace(0, 60, N)
Y = np.linspace(0, 60, N)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15, 0.2)
ax.set_zticks(np.linspace(0, 0.2, 5))
ax.view_init(27, -21)

plt.show()






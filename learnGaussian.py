import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def numpy_version():
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(0, 60, N)
    Y = np.linspace(0, 60, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([30., 31.])
    Sigma = np.array([[ 20. , 1.5], [0,  15]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = 10*multivariate_gaussian(pos, mu, Sigma)
    print(Z.shape)
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(27, -21)

    plt.show()


def torch_version():
    import torch
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(0, 60, N)
    Y = np.linspace(0, 60, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([30., 31.])
    Sigma = np.array([[20., 1.5], [0, 15]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    mu_torch = torch.Tensor(mu)
    sigma_torch = torch.Tensor(Sigma)
    pos_torch = torch.Tensor(pos)

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        pi = 3.14159265359
        n = mu.shape[0]
        assert n == 2

        Sigma_det = torch.det(Sigma)
        Sigma_inv = torch.inverse(Sigma)
        N = torch.sqrt((2*pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = torch.einsum('...k,kl,...l->...', [pos-mu, Sigma_inv, pos-mu])

        return torch.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = 10 * multivariate_gaussian(pos_torch, mu_torch, sigma_torch)
    print(Z)
    Z = Z.data.numpy()
    # Create a surface plot and projected filled contour plot under it.
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


if __name__ == '__main__':
    # numpy_version()
    torch_version()
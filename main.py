import numpy as np
import matplotlib.pyplot as plt
import torch

from mdn import MDN

# global variables
n_samples = 1000
tes_samples = 1000
n_gaussian = 5

def singleGaussian(x, mu, sigma):
    a = (2*np.pi*sigma)**0.5
    b = np.exp(-1*(x-mu)**2/(2*sigma))
    return b/a


def mdn_loss(y, mu, sigma, pi):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)


def main():
    # paraper the data

    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(-10, 10, n_samples)
    y_data = 7 * torch.sin(0.75 * x_data) + 0.5 * x_data + 1.*epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)  # inverse the x and y

    x_test = torch.linspace(-15, 15, tes_samples).view(-1, 1)

    model = MDN(n_hidden=20, n_gaussian=n_gaussian)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(15000):
        pi, mu, sigma = model(x_data)
        loss = mdn_loss(y_data, mu, sigma, pi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(epoch, loss.data.tolist())
            validate(model, x_test, x_data, y_data, epoch//500, loss.data.numpy(),
                     savefig=True)


def validate(model, x_test, x_data, y_data, epoch, loss, savefig=False):
    model.eval()
    pi, mu, sigma = model(x_test)
    # print(pi, mu, sigma)

    k = torch.multinomial(pi, 1).view(-1)  # select one with probability
    y_pred = torch.normal(mu, sigma)[np.arange(tes_samples), k].data
    # print(k)

    # plot the figures
    color_list = ['r', 'b', 'g', 'k', 'y']
    plt.figure(figsize=(12, 7))
    plt.subplot(231)
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.xlabel('Training Data', fontsize=15)

    plt.subplot(232)
    plt.title('Epoch: %.1fk, Loss:%.4f' % (epoch / 2., loss), fontsize=16)
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.xlabel('Predicted Results', fontsize=15)

    plt.subplot(233)
    for i in range(n_gaussian):
        plt.plot(30.*np.arange(tes_samples)/float(tes_samples)-15.,
                 pi.data.numpy()[:, i], color=color_list[i])
    plt.xlabel(r'$\alpha(i)$', fontsize=15)

    xx = np.linspace(-15, 15, n_samples)

    plt.subplot(234)
    pi, mu, sigma = model(torch.Tensor([-10.]))
    for i in range(n_gaussian):
        p = pi.data.numpy()[i]
        m = mu.data.numpy()[i]
        s = sigma.data.numpy()[i]
        out = singleGaussian(xx, m, s)
        plt.plot(xx, out, color=color_list[i])
    plt.xlabel('x=-10', fontsize=15)

    plt.subplot(235)
    pi, mu, sigma = model(torch.Tensor([0.]))
    for i in range(n_gaussian):
        p = pi.data.numpy()[i]
        m = mu.data.numpy()[i]
        s = sigma.data.numpy()[i]
        out = singleGaussian(xx, m, s)
        plt.plot(xx, out, color=color_list[i])
    plt.xlabel('x=0', fontsize=15)

    plt.subplot(236)
    pi, mu, sigma = model(torch.Tensor([10.]))
    for i in range(n_gaussian):
        p = pi.data.numpy()[i]
        m = mu.data.numpy()[i]
        s = sigma.data.numpy()[i]
        out = singleGaussian(xx, m, s)
        plt.plot(xx, out, color=color_list[i])
    plt.xlabel('x=10', fontsize=15)

    plt.tight_layout()
    if savefig:
        plt.savefig('epoch:%.1fk.png' % (epoch/2.), dpi=200)
        plt.close()
    # plt.show()





if __name__=='__main__':
    main()
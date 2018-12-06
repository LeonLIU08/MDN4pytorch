import numpy as np
import matplotlib.pyplot as plt
import torch

from mdn import MDN


def mdn_loss(y, mu, sigma, pi):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)


def main():
    # paraper the data
    n_samples = 1000
    tes_samples = 1000
    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(-10, 10, n_samples)
    y_data = 7 * torch.sin(0.75 * x_data) + 0.5 * x_data + 1.*epsilon

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)  # inverse the x and y

    x_test = torch.linspace(-15, 15, tes_samples).view(-1, 1)

    model = MDN(n_hidden=20, n_gaussian=5)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10000):
        pi, mu, sigma = model(x_data)
        loss = mdn_loss(y_data, mu, sigma, pi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(loss.data.tolist())

    model.eval()
    pi, mu, sigma = model(x_test)
    # print(pi, mu, sigma)

    k = torch.multinomial(pi, 1).view(-1)  # select one with probability
    y_pred = torch.normal(mu, sigma)[np.arange(tes_samples), k].data
    # print(k)
    plt.figure(figsize=(14, 8))
    plt.subplot(121)
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.subplot(122)
    for i in range(5):
        plt.plot(pi.data.numpy()[:, i])

    plt.show()

    # run it again
    pi, mu, sigma = model(x_test)
    # print(pi, mu, sigma)

    k = torch.multinomial(pi, 1).view(-1)  # select one with probability
    y_pred = torch.normal(mu, sigma)[np.arange(tes_samples), k].data
    # print(k)
    plt.figure(figsize=(14, 8))
    plt.subplot(121)
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.subplot(122)
    for i in range(5):
        plt.plot(pi.data.numpy()[:, i])

    plt.show()
















if __name__=='__main__':
    main()
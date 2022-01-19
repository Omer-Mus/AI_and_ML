# Author: Omer Mustel

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# xmin, xmax = ax.get_xlim3d()
# ymin, ymax = ax.get_ylim3d()
#
# X, Y = np.meshgrid(xrange, yrange)
# normX = normalize_arr(X)
# normX_arr = np.array(normX)
# normY = normalize_arr(Y)
# normY_arr = np.array(normY)
#
# z = betas[0] + betas[1] * normX_arr + betas[2] * normY_arr
# ax.plot_surface(X, Y, z, alpha=0.2)
# ax = plt.axes(projection ="3d")
# ax.scatter3D(x, y, z, color = "blue"



# axes3 = np.array([lin_reg_weights[0] +
                      # lin_reg_weights[1] * f1 +
                      # lin_reg_weights[2] * f2  # height
                      # for f1, f2 in zip(normalize_df(axes1), normalize_df(axes2))])



def visualize_3d(df, lin_reg_weights, xlim, ylim, zlim, feat1=0, feat2=1, labels=2,
                 alpha=200, xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
    # Setup 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Add scatter plot
    ax.scatter(df.iloc[:,feat1], df.iloc[:,feat2], df.iloc[:,labels])

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array([lin_reg_weights[0] +
                      lin_reg_weights[1] * f1 +
                      lin_reg_weights[2] * f2  # height
                      for f1, f2 in zip(axes1, axes2)])
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    title = 'Linear Regression Alpha %f' % alpha
    ax.set_title(title)

    plt.show()


def plot_alpha(i, loss, alpha, iters):
    fig = plt.figure(num=1, figsize=(10, 8))
    ax = fig.add_subplot(5, 2, i + 1)
    ax.plot(range(iters), loss, '-')
    ax.set_title(f'alpha {alpha}')


def gradient_decent(alpha, iters, X, y, n, output):
    beta = np.zeros(3)
    loss = []
    for i in range(iters):
        beta = beta - alpha * (1.0 / n) * np.transpose(X).dot(X.dot(beta) - y)
        loss_it = (1.0 / 2 * n) * np.sum(np.square(X.dot(beta) - y))
        loss.append(loss_it)
    output.append([alpha, iters, f"{beta[0]:0.8f}", f"{beta[1]:0.8f}", f"{beta[2]:0.8f}"])
    return beta, loss


def main():

    file_in, file_out = sys.argv[1], sys.argv[2]
    data = pd.read_csv(file_in, header=None)

    # Normalization
    for i in range(2):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
        # print(data[i])
    # print(list(data.columns[0:2]))
    X = np.insert(data[list(data.columns[0:2])].to_numpy(), 0, 1, axis=1)

    y = data[list(data.columns[2:])].to_numpy().flatten()
    n = len(X)

    output = []

    for i, alpha in enumerate(alphas):
        loss = gradient_decent(alpha, 100, X, y, n, output)
        # plot_alpha(i, loss[1], alpha, 100)
    my_beta = gradient_decent(0.11, 80, X, y, n, output)
    # plot_alpha(9, my_beta[1], 1, 80)
    # print(my_beta)
    pd.DataFrame(output).to_csv(file_out, header=False, index=False)

    # plt.tight_layout()
    # plt.show()
    # visualize_3d(data, lin_reg_weights=my_beta[0], xlim=(min(list(data.iloc[:, 0])), \
    #              max(list(data.iloc[:, 0]))), ylim=(min(list(data.iloc[:, 1])), max(list(data.iloc[:, 1]))), \
    #              zlim=(min(list(data.iloc[:, 2])), max(list(data.iloc[:, 2]))), alpha=0.11)


if __name__ == '__main__':
    main()

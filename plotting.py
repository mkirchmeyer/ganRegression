import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import kde
from os.path import basename
import pandas as pd


# GAN plotting

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def plots(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true, fig_dir="", save_fig=False):
    plt.plot(d_loss_err, label="Discriminator Loss")
    plt.plot(d_loss_true, label="Discriminator Loss - True")
    plt.plot(d_loss_fake, label="Discriminator Loss - Fake")
    plt.plot(g_loss_err, label="Generator Loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Loss")
    if save_fig:
        plt.savefig(f"{fig_dir}/gan_loss.png")
    plt.show()

    plt.plot(g_pred, label="Average Generator Prediction")
    plt.plot(g_true, label="Average Generator Reality")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Average Prediction")

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}gan_ave_pred.png")
    plt.show()


# Plot data

def plot_dataset(X_train, X_test, X_valid, y_train, y_test, y_valid, exp_config, fig_dir):
    plt.plot(X_train, y_train, 'rx', label="train")
    plt.plot(X_test, y_test, 'bx', label="test")
    plt.plot(X_valid, y_valid, 'gx', label="val")
    plt.title(f"Data scenario {exp_config.dataset.scenario}")
    plt.legend(loc='upper left')
    if exp_config.run.save_fig:
        plt.savefig(f"{fig_dir}/data.png")
    plt.show()


def plot_ypred_joint(x, ytrue, ypred, ypred2, ypred3, title="", alpha=0.5, prefix="0", is_sample=False, fig_dir="",
                     save_fig=False, legend=False, ylim=None, show=True):
    if ytrue is not None:
        if legend:
            plt.plot(x, ytrue, alpha=alpha, color="g", linestyle="None", marker=".", label="True")
        else:
            plt.plot(x, ytrue, alpha=alpha, color="g", linestyle="None", marker=".")

    if ypred is not None:
        if is_sample:
            plt.plot(x, ypred.tolist(), alpha=alpha, color="r", linestyle="None", marker=".", label="Linear")
        else:
            plt.plot(x, ypred[:, 0].tolist(), alpha=alpha, color="r", linestyle="None", marker=".", label="Linear")
    if ypred2 is not None:
        if is_sample:
            plt.plot(x, ypred2.tolist(), alpha=alpha, color="b", linestyle="None", marker=".")
        else:
            plt.plot(x, ypred2[:, 0].tolist(), alpha=alpha, color="b", linestyle="None", marker=".", label="GAN")
    if ypred3 is not None:
        if is_sample:
            plt.plot(x, ypred3.tolist(), alpha=alpha, color="orange", linestyle="None", marker=".")
        else:
            plt.plot(x, ypred3[:, 0].tolist(), alpha=alpha, color="orange", linestyle="None", marker=".",
                     label="GP")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

    if ylim is not None:
        plt.ylim(ylim)

    if legend:
        plt.legend(loc="upper left")
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}{prefix}_ypred.png")
    if show:
        plt.show()


# Data Distribution between Y-True and Y-Prediction.

def plot_datadistrib_joint(y, y_pred, y_pred2, y_pred3, title, width=0.1, prefix="0", fig_dir="", save_fig=False):
    max_y = max(np.max(y), np.max(y_pred))
    min_y = min(np.min(y), np.min(y_pred))
    intervals = int((max_y - min_y) / width)
    bins = np.linspace(min_y, max_y, intervals)
    plt.clf()
    if y is not None:
        plt.hist(y, alpha=0.5, color="g", label="True", bins=bins)
    if y_pred is not None:
        plt.hist(y_pred, alpha=0.5, color="r", label="LR", bins=bins)
    if y_pred2 is not None:
        plt.hist(y_pred2, alpha=0.5, color="b", label="GAN", bins=bins)
    if y_pred3 is not None:
        plt.hist(y_pred3, alpha=0.5, color="orange", label="GP", bins=bins)
    plt.legend(loc="upper right")
    plt.title(title)

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}{prefix}_hist.png")
    plt.show()


def plot_density_cont(x, y, title="", n_bins=200, ylim_min=0, y_lim_max=0, prefix="0", fig_dir="", save_fig=False):
    x = x.flatten()
    y = y.flatten()
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():n_bins * 1j, y.min():y.max():n_bins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    if ylim_min or y_lim_max:
        plt.ylim(ylim_min, y_lim_max)
    plt.title(title)

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}{prefix}_contours.png")
    plt.show()


def plot_densities_joint(ytrue, ypred, ypred2, ypred3, title, at_x=False, prefix="0", fig_dir="", save_fig=False,
                         ylim=None):
    if ytrue is not None:
        if not at_x:
            ytest_ = ytrue[:, 0].tolist()
        else:
            ytest_ = ytrue.tolist()
        ytest_ = np.sort(ytest_)
        density = stats.kde.gaussian_kde(ytest_)

    if ypred is not None:
        if not at_x:
            ypred_ = ypred[:, 0].tolist()
        else:
            ypred_ = ypred.tolist()
        ypred_ = np.sort(ypred_)
        density_pred = stats.kde.gaussian_kde(ypred_)

    if ypred2 is not None:
        if not at_x:
            ypred_2 = ypred2[:, 0].tolist()
        else:
            ypred_2 = ypred2.tolist()
        ypred_2 = np.sort(ypred_2)
        density_pred2 = stats.kde.gaussian_kde(ypred_2)

    if ypred3 is not None:
        if not at_x:
            ypred_3 = ypred3[:, 0].tolist()
        else:
            ypred_3 = ypred3.tolist()
        ypred_3 = np.sort(ypred_3)
        density_pred3 = stats.kde.gaussian_kde(ypred_3)

    if ytrue is not None:
        plt.plot(ytest_, density(ytest_), color="g", label="True")
    if ypred is not None:
        plt.plot(ypred_, density_pred(ypred_), color="r", label="Linear")
    if ypred2 is not None:
        plt.plot(ypred_2, density_pred2(ypred_2), color="b", label="GAN")
    if ypred3 is not None:
        plt.plot(ypred_3, density_pred3(ypred_3), color="orange", label="GP")

    plt.xlabel("Y")
    plt.ylabel("Probability Density")
    plt.legend(loc="upper right")
    plt.title(title)
    if ylim is not None:
        plt.xlim(ylim)

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}{prefix}_density.png")
    plt.show()

import numpy as np
from scipy.stats import multivariate_normal


def gaussian_NLPD(y_real, y_pred, cov, title=""):
    nll = -np.mean(
        [multivariate_normal.logpdf(x=y_real[i], mean=y_pred[i], cov=cov[i]) for i in range(len(y_pred))])
    print(f"{title} Gaussian NLPD: {nll}")
    return nll


def Parzen_NLPD(yTrue, yPred, bw):
    n_instances = yTrue.shape[0]
    nlpd = np.zeros((n_instances))
    for i in range(n_instances):
        n_samples = yPred[i].shape[0]
        yt = np.tile(yTrue[i], n_samples)

        E = -0.5 * np.power((yt - yPred[i].flatten()) / bw, 2)

        max_exp = np.max(E, axis=-1, keepdims=True)

        max_exp_rep = np.tile(max_exp, n_samples)
        exp_ = np.exp(E - max_exp_rep)

        constant = 0.5 * np.log(2 * np.pi) + np.log(n_samples * bw)
        nlpd[i] = -np.log(np.sum(exp_)) - max_exp + constant
    return np.mean(nlpd)


def Parzen(cgan, x, y, n_sample=10, n_bands=100):
    n_instance = x.shape[0]
    ypred_list = []
    for i in range(n_instance):
        x_ = np.tile(x[i], (n_sample, 1))
        ypred_ = cgan.predict(x_)
        ypred_list.append(ypred_)
    return min_Parzen_NLPD(y, np.array(ypred_list), n_bands)


def min_Parzen_NLPD(yTrue, yPred, n_bands=100):
    windows = np.linspace(0.01, 5, n_bands)
    nlpd = []
    for bw in windows:
        nlpd.append(Parzen_NLPD(yTrue, yPred, bw))
    inx = np.argmin(np.asarray(nlpd))
    return nlpd[inx], windows[inx], nlpd


def Parzen_test(cgan, X, y, bw, n_sample=10):
    n_instance = X.shape[0]
    ypred_list = []
    for i in range(n_instance):
        x_ = np.tile(X[i], (n_sample, 1))
        ypred_ = cgan.predict(x_)
        ypred_list.append(ypred_)
    return Parzen_NLPD(y, np.array(ypred_list), bw)

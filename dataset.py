import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def gen_data_linear(n_instance):
    a = np.random.normal(3, 3, n_instance)
    samples = int(n_instance / 2)
    X = np.hstack((np.random.normal(4, 3, samples), np.random.normal(4, 3, samples)))
    y = np.hstack((X[:samples] + a[:samples], X[samples:] + a[samples:]))
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_heteroscedastic(n_instance):
    X = np.random.normal(0, 1, n_instance)
    b = (0.001 + 0.5 * np.abs(X)) * np.random.normal(1, 1, n_instance)
    y = X + b
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_multimodal(n_instance):
    x = np.random.rand(int(n_instance / 2), 1)
    y1 = np.ones((int(n_instance / 2), 1))
    y2 = np.ones((int(n_instance / 2), 1))
    y1[x < 0.4] = 1.2 * x[x < 0.4] + 0.2 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y2[x < 0.4] = x[x < 0.4] + 0.6 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y1[np.logical_and(x >= 0.4, x < 0.6)] = 0.5 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y2[np.logical_and(x >= 0.4, x < 0.6)] = 0.6 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y1[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y2[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y = np.array(np.vstack([y1, y2])[:, 0]).reshape((n_instance, 1))
    x = np.tile(x, (2, 1)) + 0.02 * np.random.randn(n_instance, 1)
    x = np.array(x[:, 0]).reshape((n_instance, 1))

    return x, y


def gen_data_exp(n_instance):
    z = np.random.normal(0, 1, n_instance)
    X = np.random.normal(0, 1, n_instance)
    y = X + np.exp(z)
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def get_dataset(n_instance=1000, scenario="linear", seed=1):
    """
    Create regression data: y = x(1 + f(z)) + g(z)
    """

    if scenario == "linear":
        X_train, y_train = gen_data_linear(n_instance)
        X_test, y_test = gen_data_linear(n_instance)
        X_valid, y_valid = gen_data_linear(n_instance)

    elif scenario == "sinus":
        noise = 0.4
        X_full = np.linspace(start=-4, stop=4, num=3 * n_instance).reshape(-1, 1)
        np.random.shuffle(X_full)

        X_train = X_full[:n_instance]
        y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

        X_test = X_full[n_instance:2 * n_instance]
        y_test = np.sin(X_test) + noise * np.random.randn(*X_test.shape)

        X_valid = X_full[2 * n_instance:]
        y_valid = np.sin(X_valid) + noise * np.random.randn(*X_valid.shape)

    elif scenario == "heteroscedastic":
        X_train, y_train = gen_data_heteroscedastic(n_instance)
        X_test, y_test = gen_data_heteroscedastic(n_instance)
        X_valid, y_valid = gen_data_heteroscedastic(n_instance)

    elif scenario == "multi-modal":
        X_train, y_train = gen_data_multimodal(n_instance)
        X_test, y_test = gen_data_multimodal(n_instance)
        X_valid, y_valid = gen_data_multimodal(n_instance)

    elif scenario == "exp":
        X_train, y_train = gen_data_exp(n_instance)
        X_test, y_test = gen_data_exp(n_instance)
        X_valid, y_valid = gen_data_exp(n_instance)

    elif scenario == "CA-housing":
        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=seed)

    elif scenario == "CA-housing-single":
        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data[:, 0], housing.target,
                                                                      random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=seed)

        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        X_valid = X_valid.reshape(-1, 1)

    elif scenario == "ailerons":
        scaling = 1000

        my_data_train = np.genfromtxt(f"../data/Ailerons/ailerons.data", delimiter=',')
        X_train = my_data_train[:, 0:40]
        y_train = my_data_train[:, 40] * scaling

        my_data_test = np.genfromtxt(f"../data/Ailerons/ailerons.test", delimiter=',')
        X_test_full = my_data_test[:, 0:40]
        y_test_full = my_data_test[:, 40] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)
        
    elif scenario == "comp-activ":
        scaling = 1

        my_data_train = np.genfromtxt(f"../data/comp-activ/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:4096, 0:21]
        y_train = my_data_train[0:4096, 21] * scaling

        X_test_full = my_data_train[4096:8192, 0:21]
        y_test_full = my_data_train[4096:8192, 21] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "pumadyn":
        scaling = 1000
        my_data_train = np.genfromtxt(f"../data/pumadyn-32nm/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:4096, 0:32]
        y_train = my_data_train[0:4096, 32] * scaling

        X_test_full = my_data_train[4096:8192, 0:32]
        y_test_full = my_data_train[4096:8192, 32] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "bank":
        scaling = 10
        my_data_train = np.genfromtxt(f"../data/bank-32nm/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:4096, 0:32]
        y_train = my_data_train[0:4096, 32] * scaling

        X_test_full = my_data_train[4096:8192, 0:32]
        y_test_full = my_data_train[4096:8192, 32] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "census-house":
        scaling = 10**-5
        my_data_train = np.genfromtxt(f"../data/census-house/house-price-16H/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:11392, 0:16]
        y_train = my_data_train[0:11392, 16] * scaling

        X_test_full = my_data_train[11392:22784, 0:16]
        y_test_full = my_data_train[11392:22784, 16] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "abalone":
        scaling = 1
        my_data_train = np.genfromtxt(f"../data/abalone/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:2089, 1:8]
        y_train = my_data_train[0:2089, 8] * scaling

        X_test_full = my_data_train[2089:4177, 1:8]
        y_test_full = my_data_train[2089:4177, 8] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, y_train, X_test, y_test, X_valid, y_valid

import random
import numpy as np
from math import log, exp
from scipy.optimize import fmin_l_bfgs_b


def load(name):
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    return 1.0 / (1.0 + exp(-np.dot(x, theta)))


def cost(theta, X, y, lambda_):
    ret = 0

    for i in range(X.shape[0]):
        h_xi = h(X[i, :], theta)
        ret += log(h_xi) if y[i] == 1 else log(1 - h_xi)

    return - ret / X.shape[0] + lambda_ * sum(x ** 2 for x in theta)


def grad(theta, X, y, lambda_):
    M = X.shape[0]; N = X.shape[1]
    ret = [2 * x * lambda_ for x in theta]

    for i in range(M):
        f = y[i] - h(X[i, :], theta)

        for j in range(N):
            ret[j] += (-1.0 / M) * f * X[i, j]

    return np.array(ret)


def num_grad(theta, X, y, lambda_):
    EPS = 2 * 1e-4
    N = X.shape[1]
    ret = [0 for _ in range(N)]

    theta_ = theta.copy()
    for i in range(N):
        theta_[i] += EPS / 2
        fp = cost(theta_, X, y, lambda_)
        theta_[i] -= EPS
        fn = cost(theta_, X, y, lambda_)
        ret[i] = (fp - fn) / EPS
        theta_[i] = theta[i]

    return np.array(ret)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)
        return [1 - p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        X = np.hstack((np.ones((len(X),1)), X))

        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    c = learner(X, y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k = 5):
    M = X.shape[0]; N = X.shape[1]


    pr = [0 for _ in range(M)]
    iv = [0 for _ in range(M)]

    for i in range(M):
        pr[i] = i

    random.shuffle(pr)

    for i in range(M):
        iv[pr[i]] = i

    nX = [[0 for _ in range(N)] for __ in range(M)]
    ny = [0 for _ in range(M)]

    for i in range(M):
        ny[i] = y[pr[i]]

        for j in range(N):
            nX[i][j] = X[pr[i], j]


    q = M / k; r = M % k
    cum = [int(0) for _ in range(k + 1)]

    for i in range(k):
        cum[i + 1] = int(cum[i] + q + (i < r))


    ret = [[0, 0] for _ in range(M)]
    for i in range(k):
        pX = np.array(nX[cum[i] : cum[i + 1]])
        tX = np.array(nX[0 : cum[i]] + nX[cum[i + 1] : M])
        ty = np.array(ny[0 : cum[i]] + ny[cum[i + 1] : M])

        classifier = learner(tX, ty)

        for j in range(cum[i], cum[i + 1]):
            ret[j] = classifier(pX[j - cum[i]])

    retx = [[0, 0] for _ in range(M)]
    for i in range(M):
        retx[i][0] = ret[iv[i]][0]
        retx[i][1] = ret[iv[i]][1]

    return retx


def CA(real, predictions):
    assert len(real) == len(predictions)
    return sum(real[i] == (predictions[i][1] > predictions[i][0]) for i in range(len(real))) / len(real)


def CM(real, pred, threshold):
    assert len(real) == len(pred)

    ret = [[0, 0], [0, 0]]
    for i in range(len(real)):
        ret[int(real[i] >= threshold)][int(pred[i][1] >= threshold)] += 1

    return ret


def mdiv(x, y):
    return 1 if y == 0 else x / y


def AUC(real, predictions):
    predictions = np.array(predictions)
    learner = LogRegLearner(lambda_ = 0.0)
    classifier = learner(predictions, real)
    pred = [classifier(predictions[i]) for i in range(len(predictions))]

    max_it = int(1e2); EPS = 0; dE = 1 / max_it
    data = [(0, 0) for _ in range(max_it)]
    for i in range(max_it):
        conf_mat = CM(real, pred, EPS)
        data[i] = mdiv(conf_mat[0][1], conf_mat[0][0] + conf_mat[0][1]), mdiv(conf_mat[1][1], conf_mat[1][0] + conf_mat[1][1])
        EPS += dE

    ret = 0
    data = sorted(data)

    for i in range(len(data) - 1):
        x1, y1 = data[i]
        x2, y2 = data[i + 1]

        ret += (x2 - x1) * (y2 - y1) / 2
        ret += y1 * (x2 - x1)

    return ret


if __name__ == "__main__":
    X, y = load('reg.data')
    learner = LogRegLearner(lambda_=0.0)
    print("Tocnost:", CA(y, test_cv(learner, X, y)))
    print("Tocnost:", CA(y, test_learning(learner, X, y)))
from vdpgmm import VDPGMM
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import scipy as sp

def getXY(dataset = 'boston'):
    X = None
    Y = None
    if dataset == 'boston':
        boston = datasets.load_boston()
        X = boston.data
        Y = boston.target
    elif dataset == 'diabetes':
        ds = datasets.load_diabetes()
        X = ds.data
        Y = ds.target
    elif dataset == 'iris':
        ds = datasets.load_iris()
        X = ds.data
        Y = ds.target
    elif dataset == 'digits':
        ds = datasets.load_digits()
        X = ds.data
        Y = ds.target
    return X, Y

def test1():
    print 'test1'
    model = VDPGMM(T = 10, alpha = 1, max_iter = 50)
    X, Y = getXY('iris')
    model.fit(X)
    y = model.predict(X)
    print 'VDPGMM'
    print len(np.unique(y)), np.unique(y)
    print [np.sum(y == label) for label in np.unique(y)]

    from sklearn.mixture import DPGMM
    model = DPGMM(n_components = 10, alpha = 1, n_iter = 50)
    model.fit(X)
    y = model.predict(X)
    print 'DPGMM'
    print len(np.unique(y)), np.unique(y)
    print [np.sum(y == label) for label in np.unique(y)]

def test2():
    print 'test2'
    np.random.seed(1)
    X = np.concatenate((2 + np.random.randn(100, 2), 5 + np.random.randn(100, 2),  10 + np.random.randn(100, 2)))
    T = 10
    model = VDPGMM(T=T, alpha=.5, max_iter=100, thresh=1e-5)
    model.fit(X)
    
    plt.clf()
    h = plt.subplot()
    color = 'rgbcmykw'
    k = 0
    clusters = np.argmax(model.phi, axis=0)
    for t in range(T):
            xt = X[clusters == t, :]
            nt = xt.shape[0]
            if nt != 0:
                print t, nt, model.mean_mu[t,:]
                ell = mpl.patches.Ellipse(model.mean_mu[t,:], 1, 1, 50, color = color[k])
                ell.set_alpha(0.5)
                plt.scatter(xt[:, 0], xt[:, 1], color = color[k])
                h.add_artist(ell)
                k += 1

    plt.show()

def _main():
    test1()

    test2()

if __name__ == '__main__':
    _main()
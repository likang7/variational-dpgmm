import numpy as np
from vdpgmm import VDPGMM

class VDPGLM(VDPGMM):
    def __init__(self, T=1, max_iter=50, alpha=1, thresh=1e-3):
        super(VDPGLM, self).__init__(T, max_iter, alpha, thresh)

    def _initialize(self, X, y):
        super(VDPGMM, self)._initialize(X)
        self.y = np.asarray(y)
        T = self.T
        N, P = X.shape

        # w
        self.mean_w = np.random.normal(loc=0.0, scale=1e-3, size=(T, P))
        self.cov_w = np.empty((T, P, P))
        for t in xrange(T):
            self.cov_w[t] = np.eye(P)

        # beta
        self.a_beta = 1
        self.b_beta = np.ones(T)

        # xi
        self.a_xi = np.ones(T)
        self.b_xi = np.ones(T)
        self.xi = self.a_xi / self.b_xi

        # muy
        self.mean_muy = np.random.normal(loc=0.0, scale=1.0, size=(T))
        self.cov_muy = np.ones(T)

        self.a_beta0 = 1e-3
        self.b_beta0 = 1e-3
        self.a_xi0 = 1e-3
        self.b_xi0 = 1e-3
        self.muy_prec0 = 1

    def _update_w(self):
        pass

    def _update_beta(self):
        self.a_beta = self.a_beta0 + .5 * self.P
        for t in xrange(self.T):
            self.b_beta[t] = self.b_beta0 + .5 * (self.mean_w[t].T.dot(self.mean_w[t]) + np.trace(self.cov_w[t])) 
        self.beta = self.a_beta / self.b_beta

    def _update_xi(self):
        pass

    def _update_muy(self):
        pass

    def _update_c(self):
        pass

    def _update(self, X, y):
        super(VDPGMM, self)._update(X)

        self._update_muy()
        self._update_beta()
        self._update_w()
        self._update_xi()

    def fit(self, X, y):
        self._initialize(X, y)

        update_func = lambda: self._update(self.X, self.y)

        self._do_fit(update_func)

    def predict(self, x):
        x = np.asarray(x)
        phi = super(VDPGMM, self)._update_c(x)
        y = np.zeros(x.shape[0])
        for t in xrange(self.T):
            y += phi[t] * (x.dot(self.mean_w[t]) + self.mean_muy[t])
        return y

    def lowerbound(self):
        lb = super(VDPGMM, self).lowerbound()

        return lb

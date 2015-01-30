import numpy as np
from vdpgmm import VDPGMM
from numpy.linalg import inv
from scipy.special import digamma, gammaln

class VDPGLM(VDPGMM):
    def __init__(self, T=1, max_iter=50, alpha=1, thresh=1e-3):
        super(VDPGLM, self).__init__(T, max_iter, alpha, thresh)

    def _initialize(self, X, y):
        super(VDPGLM, self)._initialize(X)
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

        self.bound_Y = self._bound_y(X, y)

    def _update_w(self, X, y):
        for t in xrange(self.T):
            q_xx = np.multiply(self.phi[t], X.T).dot(X)
            self.cov_w[t] = inv(self.beta[t] * np.eye(self.P) + self.xi[t]*q_xx)
            w = np.multiply(y - self.mean_muy[t], self.phi[t])
            self.mean_w[t] = self.xi[t] * self.cov_w[t].dot(X.T.dot(w))

    def _update_beta(self):
        self.a_beta = self.a_beta0 + .5 * self.P
        for t in xrange(self.T):
            self.b_beta[t] = self.b_beta0 + .5 * (self.mean_w[t].T.dot(self.mean_w[t]) + np.trace(self.cov_w[t])) 
        self.beta = self.a_beta / self.b_beta

    def _update_xi(self, bound_Y):
        self.a_xi = self.a_xi0 + .5 * self.Nt
        self.b_xi = self.b_xi0 + .5 * np.sum(self.phi * bound_Y, axis=1)

    def _update_muy(self, X, y):
        self.cov_muy = (1.0 * self.muy_prec0 + self.xi * self.Nt) ** -1

        for t in xrange(self.T):
            m = self.phi[t].dot(y - X.dot(self.mean_w[t]))
            self.mean_muy[t] = self.xi[t] * self.cov_muy[t] * m

    def _log_lik_y(self, bound_Y):
        liky = np.empty(bound_Y.shape)
        for t in xrange(self.T):
            liky[t, :] = .5 * (digamma(self.a_xi[t]) - np.log(self.b_xi[t]) - np.log(2*np.pi))
            liky[t] -= .5 * self.xi[t] * bound_Y[t]
        return liky

    def _update_c(self, X, bound_Y=None):
        bound_Y = self.bound_Y if bound_Y is None else bound_Y

        likc = self._log_lik_pi()

        likx = self._log_lik_x(X)

        liky = self._log_lik_y(bound_Y)

        s = likc[:, np.newaxis] + likx + liky

        phi = self._log_normalize(s, axis=0)

        return phi, np.sum(self.phi, axis = 1)

    def _bound_y(self, X, y):
        bound_y = np.empty((self.T, y.shape[0]))
        for t in xrange(self.T):
            np.square(y - self.mean_muy[t] - X.dot(self.mean_w[t]), out=bound_y[t])
            bound_y[t] += self.cov_muy[t]
            for n in xrange(y.shape[0]):
                bound_y[t, n] += np.sum(np.outer(X[n], X[n])*self.cov_w[t])
        return bound_y

    def _update(self, X, y):
        super(VDPGLM, self)._update(X)

        self._update_muy(X, y)
        self._update_beta()
        self._update_w(X, y)

        self.bound_Y = self._bound_y(X, y)
        self._update_xi(self.bound_Y)

    def fit(self, X, y):
        self._initialize(X, y)

        update_func = lambda: self._update(self.X, self.y)

        self._do_fit(update_func)

    def predict(self, x):
        x = np.asarray(x)
        phi, _ = super(VDPGLM, self)._update_c(x)
        y = np.zeros(x.shape[0])
        for t in xrange(self.T):
            y += phi[t] * (x.dot(self.mean_w[t]) + self.mean_muy[t])
        return y

    def lowerbound(self):
        lb = super(VDPGLM, self).lowerbound()

        T = self.T
        N, P = self.X.shape
        K = self.K

        #beta
        d_a_beta = digamma(self.a_beta)
        ln_b_beta = np.log(self.b_beta)
        ln_beta = d_a_beta - ln_b_beta
        lpbeta = np.sum(-gammaln(self.a_beta0) + self.a_beta0*np.log(self.b_beta0) \
            + (self.a_beta0 - 1)*ln_beta - self.b_beta0*self.beta)
        lqbeta = np.sum(-gammaln(self.a_beta) + (self.a_beta - 1)*d_a_beta \
            + ln_b_beta - self.a_beta)
        lb += lpbeta - lqbeta

        #w
        Eq_ww = np.empty(T)
        for t in xrange(T):
            Eq_ww[t] = self.mean_w[t].dot(self.mean_w[t]) + np.trace(self.cov_w[t])
        lpw = .5 * np.sum(K*ln_beta - self.beta*Eq_ww)
        lqw = 0
        for t in xrange(T):
            sign, logdet = np.linalg.slogdet(self.cov_w[t])
            lqw += -0.5*(sign*logdet + K)
        lb += lpw - lqw

        #y
        liky = self._log_lik_y(self.bound_Y)
        lpy = np.sum(self.phi * liky)
        lb += lpy

        #xi
        d_a_xi = digamma(self.a_xi)
        ln_b_xi = np.log(self.b_xi)
        ln_xi = d_a_xi - ln_b_xi
        lpxi = np.sum(-gammaln(self.a_xi0) + self.a_xi0*np.log(self.b_xi0) \
            + (self.a_xi0-1)*ln_xi - self.b_xi0*self.xi)
        lqxi = np.sum(-gammaln(self.a_xi) + (self.a_xi-1)*d_a_xi \
            + ln_b_xi - self.a_xi)
        lb += lpxi - lqxi

        #mu_y
        lpmuy = .5 * (T * np.log(self.muy_prec0) - self.muy_prec0 * \
            (self.mean_muy.dot(self.mean_muy)+np.sum(self.cov_muy)))
        lqmuy = .5 * np.sum(np.log(self.cov_muy) + 1)
        lb += lpmuy - lqmuy
        
        return lb

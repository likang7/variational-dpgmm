import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.special import digamma, gammaln
import sys
import math

def load_data():
	import csv
	from sklearn import preprocessing

	handler = open('parkinsons.data.csv')
	D = len(handler.readline().split(',')) - 2
	reader = csv.reader(handler)
	lines = []
	for line in reader:
		lines.append(line[1:])
	data = np.array(lines)

	X = np.array(data[:, :D]).astype(float)
	Y = np.array(data[:, D]).astype(float)
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)

	return X.astype(float), Y.astype(float)

def lowerbound(x, alpha, gamma, phi, ngmu, nglambda, ngalpha, ngbeta, T):
	sumGamma = gamma[:, 0] + gamma[:, 1]
	# Eq[log V_t]
	Eq_logv1 = digamma(gamma[:, 0]) - digamma(sumGamma)
	# Eq[log (1-V_t)]
	Eq_logv2 = digamma(gamma[:, 1]) - digamma(sumGamma)

	lb = 0.
	# Eq[log p(V | 1, alpha)] ... OK
	lb += (alpha - 1) * np.sum(Eq_logv2) + (T-1) * (gammaln(1.0 + alpha) - gammaln(alpha))
	# Eq[log q(V | gamma1, gamma2)] ... OK
	lb -= np.sum(((gamma[:, 0] - 1) * Eq_logv1 + (gamma[:, 1] - 1) * Eq_logv2) + \
		gammaln(sumGamma) - gammaln(gamma[:, 0]) - gammaln(gamma[:, 1]))

	# Eq[log p(Z | V)] ... OK
	phi_cum = np.cumsum(phi[:0:-1, :], axis = 0)[::-1, :] #sum(j=t+1, T)phi[n,j]
	lb += np.sum(np.multiply(phi_cum.T, Eq_logv2) + np.multiply(phi[:-1,:].T, Eq_logv1))
	# Eq[log q(Z | phi)] ... OK
	phi_non_zeros = phi[phi > np.finfo(np.float32).eps]
	lb -= np.sum((np.multiply(phi_non_zeros, np.log(phi_non_zeros))))

	# Eq[log p(tau)] ... OK
	lb += - np.sum(np.divide(ngalpha, ngbeta))
	# Eq[log q(tau | ngalpha, ngbeta)] ... OK
	lb -= np.sum(-gammaln(ngalpha) + np.multiply(ngalpha - 1, digamma(ngalpha)) \
	 	+ np.log(ngbeta) - ngalpha)

	# Eq[log p(mu)] ... OK
	lb += -0.5 * np.sum(np.square(ngmu))
	# Eq[log q(mu | ngmu, lambda.tau)]
	lb -= 0.5 * np.sum(digamma(ngalpha) - np.log(ngbeta))

	# Eq[log p(X)]
	#lb = 0.
	N, D = x.shape
	for t in range(T):
		phi_t = phi[t, :] # 1 * N
		p_x = -0.5 * D * np.log(2 * np.pi) + 0.5 * \
			np.sum((digamma(ngalpha[t, :]) - np.log(ngbeta[t, :])) - \
			np.divide(ngalpha[t, :], ngbeta[t, :]) * \
			np.square(x - ngmu[t,:]), axis = 1) \
			- 0.5 * np.sum(ngalpha[t, :] / ngbeta[t,:]) # last line need?
		lb += np.dot(phi_t, p_x)
	return lb


def norm_pdf(x, mu, tau):
	x_mu = x - mu
	norm_const = np.power(tau, 0.5) / np.power(2 * np.pi, 0.5)
	norm_result = np.exp(-0.5 * tau * (np.square(x_mu)))
	return norm_result * norm_const

def norm_logpdf(x, mu, tau):
	x_mu2 = np.square(x-mu)
	log_const = 0.5 * (np.log(tau) - np.log(2 * np.pi))
	log_result = -0.5 * tau * x_mu2
	return log_const + log_result

def update_gamma(gamma, alpha, phi, T):
	gamma[:, 0] = 1 + np.sum(phi[:T-1], axis = 1)
	phi_cum = np.cumsum(phi[:0:-1, :], axis = 0)[::-1, :]
	gamma[:, 1] = alpha + np.sum(phi_cum, axis = 1)
	return gamma

def update_phi(phi, gamma, X, mu, tau, T):
	sumGamma = gamma[:, 0] + gamma[:, 1]
	# Eq[log V_t]
	Eq_logv1 = digamma(gamma[:, 0]) - digamma(sumGamma)
	# Eq[log (1-V_t)]
	Eq_logv2 = digamma(gamma[:, 1]) - digamma(sumGamma)

	cumsum_Eq_logv2 = np.cumsum(Eq_logv2)
	Eq_pi = np.zeros(T)
	Eq_pi[0] = np.exp(Eq_logv1[0])
	Eq_pi[1:-1] = np.exp(Eq_logv1[1:] + cumsum_Eq_logv2[0:-1])
	Eq_pi[-1] = 1. - np.sum(Eq_pi[0:-1])

	for t in range(T):
		logpdf = norm_logpdf(X, mu[t, :], tau[t, :])
	 	phi[t, :] = np.sum(logpdf, axis = 1)

	phi = np.exp(np.subtract(phi, np.max(phi, axis = 0)))
	phi = np.multiply(phi, Eq_pi.reshape(T, 1))
	phi = np.divide(phi, np.sum(phi, axis = 0))	#normalize 

	return phi

def update_alpha(alpha, s1, s2, gamma, T): 
	w1 = s1 + T - 1.
	w2 = s2 - np.sum(digamma(gamma[:, 1]) - digamma(gamma[:, 0] + gamma[0:, 1]))
	return w1 / w2

'''
alpha ~ Gamma(1, 1)
V ~ Beta(1, alpha)
Z ~ SBP(V)
(Mu, Tau) ~ NormalGamma(mu0, lambda0, 1, 1)
X ~ N(Mu, Tau^-1)

q(alpha) ~ Gamma(alpha | w1, w2)
q(vt) ~ Beta(vt | gamma_t,1, gamma_t,2)
q(zn) ~ Discrete(zn | phi_n)
q(mu_t, tau_t) ~ NormalGamma(ngmu, nglambda, ngalpha, ngbeta)
'''
def v_dpmm(X, alpha = 0.1, T = 30, n_iter = 100):
	N, D = X.shape

	# init hyperparameters
	s1 = 1
	s2 = 1

	# init latent variable
	gamma = np.zeros((T - 1, 2))
	
	phi = np.random.uniform(size = (T, N))
	phi = np.divide(phi, np.sum(phi, axis = 0))	#normalize 

	ngmu0 = np.zeros((1, D))
	nglambda0 = 1
	ngalpha0 = 1
	ngbeta0 = 1

	ngmu = np.zeros((T, D))
	nglambda = np.zeros((T, D))
	ngalpha = np.ones((T, D))
	ngbeta = np.ones((T, D))

	lbs = []
	print "training",
	#loop update
	for i in range(n_iter):
		print("."),; sys.stdout.flush()
		#update gamma
		gamma = update_gamma(gamma, alpha, phi, T)

		#update alpha
		alpha = update_alpha(alpha, s1, s2, gamma, T)

		#update ngmu, nglambda, ngalpha, ngbeta
		clusters = np.argmax(phi, axis=0)
		for t in range(T):
			xt = X[clusters == t, :]
			nt = xt.shape[0]
			if nt == 0:
				ngmu[t, :] = ngmu0
				nglambda[t] = nglambda0
				ngalpha[t] = ngalpha0
				ngbeta[t,:] = ngbeta0
				continue
			meanxt = np.mean(xt, axis = 0)
			ngmu[t, :] = (1. * nglambda0 * ngmu0 + nt * meanxt) / (nglambda0 + nt)
			nglambda[t, :] = (nglambda0 + nt) #* (ngalpha[t, :] / ngbeta[t, :])
			ngalpha[t, :] = ngalpha0 + 0.5 * nt #+ 0.5
			ngbeta[t, :] = ngbeta0 + 0.5 * (np.sum(np.square(xt - meanxt), axis = 0) \
			 + nglambda0 * nt * np.square(meanxt - ngmu0) / (nglambda0 + nt))
			# ngbeta[t, :] = ngbeta0 + 0.5 * ((nglambda0 + nt) * (1. / nglambda[t,:] + \
			# 	np.square(ngmu[t, :])) - 2. * \
			# (nglambda0 * ngmu0 + np.sum(xt, axis = 0)) * ngmu[t,:] + \
			# np.sum(np.square(xt), axis = 0) + nglambda0*ngmu0**2)

		#update phi
		phi = update_phi(phi, gamma, X, ngmu, np.divide(ngalpha, ngbeta), T)
		
		# calculate lower bound
		lb = lowerbound(X, alpha, gamma, phi, ngmu, nglambda, ngalpha, ngbeta, T)
		lbs.append(lb)
	return gamma, phi, ngmu, nglambda, ngalpha, ngbeta, lbs

import matplotlib.pylab as plt
import matplotlib as mpl
def _main():
	#X, Y = load_data()
	np.random.seed(1)
	X = np.concatenate((2 + np.random.randn(100, 2), 7 + 1 * np.random.randn(100, 2),  10 + np.random.randn(100, 2)))
	T = 50
	gamma, phi, ngmu, nglambda, ngalpha, ngbeta, lbs = v_dpmm(X, alpha=1, T=T, n_iter= 50)
	print "" 
	print lbs
	plt.clf()
	h = plt.subplot()
	color = 'rgbcmykw'
	k = 0
	clusters = np.argmax(phi, axis=0)
	for t in range(T):
			xt = X[clusters == t, :]
			nt = xt.shape[0]
			if nt != 0:
				print t, nt, ngmu[t,:], ngbeta[t,:] / ngalpha[t,:]
				ell = mpl.patches.Ellipse(ngmu[t,:], 1, 1, 50, color = color[k])
				ell.set_alpha(0.5)
				plt.scatter(xt[:, 0], xt[:, 1], color = color[k])
				h.add_artist(ell)
				k += 1

	plt.show()

	# compare with sklearn.mixture.DPGMM
	# print ""
	# from sklearn.mixture import DPGMM
	# model = DPGMM(n_components = T, alpha = 5, n_iter = 50)
	# model.fit(X)
	# y = model.predict(X)
	# for t in range(T):
	# 	xt = X[y == t, :]
	# 	nt = xt.shape[0]
	# 	if nt != 0:
	# 		print t, nt, model.means_[t]
	# print model.means_[np.unique(y)]

#load_data()
_main()

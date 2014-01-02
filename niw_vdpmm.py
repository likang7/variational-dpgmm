import numpy as np
import scipy as sp
from scipy.special import digamma, gammaln
import sys
import math

def multivariate_norm_logpdf(x, mu, sigma):
	D = x.shape[1]
	if D == mu.shape[0] and (D, D) == sigma.shape:
		det = np.linalg.det(sigma)
		if det <= 0:
			raise NameError("The covariance matrix can't be singular")

		x_mu = x - mu
		inv_sigma = np.linalg.inv(sigma)
		log_norm_const = -0.5 * math.log(det) - D / 2.0 * math.log(2*np.pi)

		log_result = -0.5*(np.dot(np.dot(x_mu, inv_sigma), x_mu.T).diagonal().T)
		return log_norm_const + log_result
	else:
		raise NameError("The dimensions of the input don't match")

def v_dpmm(X, alpha = 0.1, T = 30, n_iter = 1000):
	'''
	alpha ~ Gamma(s1, s2)
	V ~ Beta(1, alpha)
	Z ~ Discrete(pi(V))
	(Mu, Sigma) ~ N(m0, (beta0*Sigma)^-1)W(S, p)
	X ~ N(Mu, Sigma, Z)

	q(alpha) ~ Gamma(alpha | w1, w2)
	q(vt) ~ Beta(vt | gamma_t,1, gamma_t,2)
	q(zn) ~ Discrete(zn | phi_n)
	q(mu_t, sigma_t) ~ N(mu_t | m_t, (beta_t * sigma_t)^-1)W(sigma_t | s_t, p_t)
	'''
	N, D = X.shape

	# init hyperparameters
	s1 = 1
	s2 = 1
	# init latent variable
	gamma = np.ones((T - 1, 2))
	
	phi = np.random.uniform(size = (T, N))
	phi = np.divide(phi, np.sum(phi, axis = 0))	#normalize 

	m0 = np.zeros((1, D))
	beta0 = 1
	p0 = D + 2
	s0 = np.eye(D)

	m = np.random.uniform(size=(T, D))
	beta = np.ones((T, 1))
	s = np.random.uniform(size = (T, D, D))
	p = np.zeros((T, 1)) + 2

	#loop update
	for i in range(n_iter):
		print("."),; sys.stdout.flush()

		#update gamma
		gamma[:, 0] = 1 + np.sum(phi[:T-1], axis = 1)
		phi_cum = np.cumsum(phi[:0:-1, :], axis = 0)[::-1, :]
		gamma[:, 1] = alpha + np.sum(phi_cum, axis = 1)

		#update m,beta,p,s
		clusters = np.argmax(phi, axis=0)
		for t in range(T):
			xt = X[clusters == t, :]
			nt = xt.shape[0]
			if nt == 0:
				m[t, :] = m0
				beta[t] = beta0
				p[t] = p0
				s[t,:] = s0
				continue
			meanxt = np.mean(xt, axis = 0)
			m[t, :] = 1. * (nt * meanxt + beta0 * m0) / (nt + beta0)
			beta[t] = beta0 + nt
			p[t] = p0 + nt
			s[t, :] = s0 + np.dot((xt - meanxt).T, (xt - meanxt)) / float(nt) + \
				 (1. * nt * beta0 / (nt + beta0) * np.dot((meanxt - m0).T, (meanxt - m0)))
			#print t, s[t,:]

		#update phi
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


		# Eq_v = np.divide(1.0 * gamma[:, 0], gamma[:,0] + gamma[:, 1])
		# negv_cum = np.cumproduct(1 - Eq_v, axis = 0)
		# Eq_pi = np.zeros((T, 1))
		# Eq_pi[0] = Eq_v[0]

		# tmp = Eq_v * negv_cum
		# for j in range(1, T):
		# 	Eq_pi[j] = tmp[j-1]

		Eq_mu = m;
		Eq_sigma = s
		#for t in range(0, T):
		# 	Eq_sigma[t,:,:] /= p[t] - D - 1.0
		for t in range(T):
			phi[t, :] = multivariate_norm_logpdf(X, Eq_mu[t,:], Eq_sigma[t,:,:])

		phi = np.exp(np.subtract(phi, np.max(phi, axis = 0)))
		phi = np.multiply(phi, Eq_pi.reshape(T, 1))
		phi = np.divide(phi, np.sum(phi, axis = 0))	#normalize 
	return gamma, phi, m, beta, s, p


def _main():
	np.random.seed(1)
	mean1 = [1,-1]
	cov1 = [[1,1],[1,1]]
	mean2 = [10,5]
	cov2 = [[1,1],[2,1]]
	X= np.concatenate((np.random.multivariate_normal(mean1,cov1,100), \
		np.random.multivariate_normal(mean2,cov2,100)))
	
	T = 50
	gamma, phi, m, beta, s, p = v_dpmm(X, alpha=1, T=T, n_iter= 20)
	clusters = np.argmax(phi, axis=0)
	print ""
	for t in range(T):
			xt = X[clusters == t, :]
			nt = xt.shape[0]
			if nt != 0:
				print t, nt, m[t,:]
	return

_main()



Variational inference for Dirichlet Process mixture models with multivariate normal and diagonal normal mixture components. 
Based on the following paper:

Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. Bayesian analysis, 1(1), 121-143. [pdf](http://www.cs.princeton.edu/~blei/papers/BleiJordan2004.pdf)

ng_vdpmm use the model below:  
alpha ~ Gamma(s1, s2)  
V ~ Beta(1, alpha)  
Z ~ SBP(V)  
(Mu, Tau) ~ NormalGamma(mu0, lambda0, 1, 1)  
X ~ N(Mu, Tau^-1)  

q(alpha) ~ Gamma(alpha | w1, w2)  
q(vt) ~ Beta(vt | gamma_t,1, gamma_t,2)  
q(zn) ~ Discrete(zn | phi_n)  
q(mu_t, tau_t) ~ NormalGamma(ngmu, nglambda, ngalpha, ngbeta)  


niw_vdpmm use the model below:  
alpha ~ Gamma(s1, s2)  
V ~ Beta(1, alpha)  
Z ~ SBP(V)  
(Mu, Sigma) ~ N(m0, (beta0*Sigma)^-1)W(S, p)  
X ~ N(Mu, Sigma, Z)  

q(alpha) ~ Gamma(alpha | w1, w2)  
q(vt) ~ Beta(vt | gamma_t,1, gamma_t,2)  
q(zn) ~ Discrete(zn | phi_n)  
q(mu_t, sigma_t) ~ N(mu_t | m_t, (beta_t * sigma_t)^-1)W(sigma_t | s_t, p_t)

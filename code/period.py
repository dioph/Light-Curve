from scipy.optimize import leastsq, minimize
from astropy.convolution import Box1DKernel
import emcee
import celerite
from celerite import terms
from tqdm import tqdm
import autograd.numpy as np

from .lightcurve import acf, make_gauss, smooth, filt
from .utils import plot_mcmc
from .periodogram import find_peaks
from time import perf_counter


class CustomTerm(terms.Term):
    parameter_names = ("log_B", "log_C", "log_L", "log_P")
    def get_real_coefficients(self, params):
        log_B, log_C, log_L, log_P = params
        a = np.exp(log_B)
        b = np.exp(log_C)
        c = np.exp(-log_L)
        
        return (a * (1.0 + b) / (2.0 + b), c)
        
    def get_complex_coefficients(self, params):
        log_B, log_C, log_L, log_P = params
        a = np.exp(log_B)
        b = np.exp(log_C)
        c = np.exp(-log_L)
        return (a / (2.0 + b), 0.0, c, 2*np.pi*np.exp(-log_P))


class RotationModeler(object):
    def __init__(self, lc):
        lc = lc.remove_nans().remove_outliers()
        self.x = lc.time - lc.time.min()
        self.y = lc.flux/lc.flux.mean() - 1
        self.decimate()
        bounds = dict(log_B=(-20, 0), log_C=(None,None), log_L=(None,None), log_P=(-3, 5))
        self.term = terms.JitterTerm(log_sigma=-17)
        self.term += CustomTerm(log_B=-10, log_C=0, log_L=3, log_P=1, bounds=bounds)
        self.gp = celerite.GP(self.term)
        self.prior = np.ones_like
    
    def make(self, pmin=0.1):
        ti, hi, qi = peaks(self.x, self.y, pmin)
        self.prior = make_prior(ti, qi)
        
    def decimate(self, xmin=None, xmax=None, dec=None):
        if xmin is None:
            xmin = self.x.min()
        if xmax is None:
            xmax = self.x.max()
        if dec is None:
            dec = 1
        mask = np.logical_and(self.x >= xmin, self.x <= xmax)
        n1 = self.x[mask].size // dec
        self.xdec = np.array([a[0] for a in np.array_split(self.x[mask], n1)])
        self.ydec = np.array([a[0] for a in np.array_split(self.y[mask], n1)])
        
    def lnlike(self, p, fast=False):
        self.gp.set_parameter_vector(p)
        x = self.xdec
        y = self.ydec
        if fast:
            L = 300
            n2 = x.size // L
            ll = 0.0
            xs = np.array_split(x, n2)
            ys = np.array_split(y, n2)
            for a,b in zip(xs, ys):
                self.gp.compute(a)
                ll += self.gp.log_likelihood(b)
        else:
            self.gp.compute(x)
            ll = self.gp.log_likelihood(y)
        return ll
    
    def lnprior(self, p):
        s, a, b, c, P = p
        gaussians = [make_gauss(-17, 5), make_gauss(-10, 5), make_gauss(0, 5),
                     make_gauss(3, 2), self.prior]
        return np.log(gaussians[0](s) * gaussians[1](a) * gaussians[2](b)
                        * gaussians[3](c) * gaussians[4](P))
    
    def lnprob(self, p, fast=False):
        self.gp.set_parameter_vector(p)
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(p, fast)
        
    def nll(self, p, y):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(y)
        
    def grad_nll(self, p, y):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(y)[1]
        
    def minimize(self):
        x = self.xdec
        y = self.ydec
        assert self.xdec.size <= 10000, "Don't forget to decimate before minimizing! (N={})".format(self.xdec.size)
        self.gp.compute(x)
        print('Initial likelihood:', self.gp.log_likelihood(y))
        t1 = perf_counter()
        p0 = self.gp.get_parameter_vector()
        bounds = self.gp.get_parameter_bounds()
        results = minimize(self.nll, p0, jac=self.grad_nll, method='L-BFGS-B', 
                            args=(y), bounds=bounds)
        self.gp.set_parameter_vector(results.x)
        self.gp.compute(x)
        print('Final likelihood:', self.gp.log_likelihood(y))
        t = np.linspace(x.min(), x.max(), 5000)
        mu, var = self.gp.predict(y, t, return_var=True)
        std = np.sqrt(var)
        t2 = perf_counter()
        print('Done in {0:.2f} seconds.'.format(t2-t1))
        return t, mu, std, self.gp.get_parameter_vector()
        
    def mcmc(self, nwalkers=50, nsteps=1000, burn=0, fast=False):
        ndim = len(self.gp)
        prob = lambda p: self.lnprob(p, fast=fast)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob)
        p = self.gp.get_parameter_vector()
        p0 = p + 1e-5 * np.random.randn(nwalkers, ndim)
        t1 = perf_counter()
        for result in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps):
            pass
        t2 = perf_counter()
        print('Done in {0:.2f} seconds.'.format(t2-t1))
        chain = sampler.chain
        samples = chain[:, burn:, :].reshape(-1, ndim)
        return samples
        
    def plot(self, samples=None, ptrue=None, nbins=30, usecols=[0,1,2,3,4], **kwargs):
        if samples is None:
            samples = self.mcmc(**kwargs)
        names = np.array(['$\ln{\sigma_n}$', '$\ln{B}$', '$\ln{C}$', '$\ln{L}$', '$\ln{P}$'])
        gaussians = [make_gauss(-17,5), make_gauss(-10,5), make_gauss(0,5), 
                    make_gauss(3,2), self.prior]
        priors = []
        for i in usecols:
            priors.append(gaussians[i])
        plot_mcmc(samples[:, usecols], labels=names[usecols], priors=priors, 
                    ptrue=ptrue, nbins=nbins)


def peaks(t, f, pmin=0.1):
    fs = 1/np.median(t[1:] - t[:-1])
    t -= min(t)
    ts = []
    hs = []
    qs = []
    for i in range(8):
        Pi = 2**i
        if Pi >= max(t)/2:
            continue
        y = filt(f, 1/Pi, 1/pmin, fs)
        ml = np.where(t >= 2*Pi)[0][0]
        R = acf(y, maxlag=ml)
        if Pi >= 20:
            R = smooth(R, Box1DKernel(width=Pi//10))
        ti, hi = find_peaks(R, t)
        search = np.where(R < 0)[0][0]
        peak = search + np.argmax(R[search:])
        ts.append(ti)
        hs.append(hi)
        results, msg = leastsq(make_eps(ti[i], 20*Pi/ti[i]), [1, 1], args=(t[:ml], R))
        A, tau = results
        eps = make_eps(ti[i], 20*Pi/ti[i])
        ri = eps(results, t[:ml], R).sum()
        qs.append((tau/ti[i])*(ml*hi[i]/ri))
    ts = np.array(ts)
    hs = np.array(hs)
    qs = np.array(qs)
    return ts, hs, qs


def make_model_acf(T):
    def model_acf(p, x):
        A = p[0]
        tau = p[1]
        return A * np.exp(-x/tau) * np.cos(2 * np.pi * x / T)
    return model_acf    


def make_eps(T, maxt):
    def eps(p, x, y):
        mod = make_model_acf(T)
        if p[1] > maxt:
            return np.ones_like(y)
        return np.square(y - mod(p, x))
    return eps


def make_prior(t, q):
    def prior(logP):
        tot = 0
        for ti,qi in zip(t,q):
            qi = max(qi, 0)
            gaussian = make_gauss(np.log(ti), 0.2)
            tot += qi * (0.9 * gaussian(logP) +
                         0.05 * gaussian(logP) +
                         0.05 * gaussian(logP))
        tot /= q.sum()
        return tot
    return prior


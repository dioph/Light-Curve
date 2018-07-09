from scipy.optimize import leastsq, minimize, least_squares
from scipy.stats import linregress
from astropy.convolution import Box1DKernel
import emcee
import celerite
from celerite import terms
import george
from george import kernels
from tqdm import tqdm
import autograd.numpy as np

from .lightcurve import acf, smooth, make_gauss, filt
from .utils import plot_mcmc
from .periodogram import find_peaks
from time import perf_counter


class RotationModeler(object):
    def __init__(self, lc):
        lc = lc.remove_nans().remove_outliers()
        self.x = lc.time - lc.time.min()
        self.y = lc.flux
        if self.y.mean() > 0.1:
            self.y /= np.median(self.y)
        m, b = linregress(self.x, self.y)[:2]
        mean = m * self.x + b
        self.y -= mean
        self.decimate()
        self.prior = lambda p: np.array([-np.inf, 0])[np.array(np.logical_and(-0.69 < p, p < 4.61), dtype=int)]

    def make(self, pmin=0.1):
        ti, hi, qi = peaks(self.xdec, self.ydec, pmin)
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
            for a, b in zip(xs, ys):
                self.gp.compute(a)
                ll += self.gp.log_likelihood(b, quiet=True)
        else:
            self.gp.compute(x)
            ll = self.gp.log_likelihood(y, quiet=True)
        return ll

    def lnprior(self, p):
        gaussians = np.append([make_gauss(self._mu[i], self._sigma[i]) for i in
                               range(len(self._mu))], [self.prior])
        for i, (lo, hi) in enumerate(self._bounds):
            if p[i] > hi or p[i] < lo:
                return -np.inf
        lp = np.sum(np.log(gaussians[i](p[i])) for i in range(len(p)))
        return lp

    def sample_prior(self, N):
        ndim = len(self.gp)
        samples = np.inf * np.ones((N, ndim))
        m = np.ones(N, dtype=bool)
        nbad = m.sum()
        while nbad > 0:
            r = np.random.randn(N * (ndim - 1)).reshape((N, ndim - 1))
            for i in range(ndim - 1):
                samples[m, i] = r[m, i] * self._sigma[i] + self._mu[i]
            samples[m, -1] = self.sample_prots(nbad)
            lp = np.array([self.lnprior(p) for p in samples])
            m = ~np.isfinite(lp)
            nbad = m.sum()
        return samples

    def sample_prots(self, N):
        logP = np.linspace(-0.69, 4.61, 1061)
        probs = self.prior(logP)
        prots = np.random.choice(1061, N, p=probs)
        samples = logP[prots]
        return samples

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
        results = minimize(self.nll, p0, jac=self.grad_nll, method='L-BFGS-B',
                           args=(y), bounds=self._bounds)
        self.gp.set_parameter_vector(results.x)
        self.gp.compute(x)
        print('Final likelihood:', self.gp.log_likelihood(y))
        t = np.linspace(x.min(), x.max(), 5000)
        mu, var = self.gp.predict(y, t, return_var=True)
        std = np.sqrt(var)
        t2 = perf_counter()
        print('Done in {0:.2f} seconds.'.format(t2 - t1))
        return t, mu, std, self.gp.get_parameter_vector()

    def mcmc(self, nwalkers=50, nsteps=1000, burn=0, fast=False, useprior=False):
        ndim = len(self.gp)
        prob = lambda p: self.lnprob(p, fast=fast)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob)
        p = self.gp.get_parameter_vector()
        if useprior:
            p0 = self.sample_prior(nwalkers)
        else:
            p0 = p + 1e-5 * np.random.randn(nwalkers, ndim)
        for result in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps):
            pass
        chain = sampler.chain
        samples = chain[:, burn:, :].reshape(-1, ndim)
        return samples

    def plot(self, samples=None, ptrue=None, nbins=30, usecols=[0, 1, 2, 3, 4], **kwargs):
        if samples is None:
            samples = self.mcmc(**kwargs)
        gaussians = np.append([make_gauss(self._mu[i], self._sigma[i]) for i in
                               range(len(self._mu))], [self.prior])
        priors = []
        for i in usecols:
            priors.append(gaussians[i])
        plot_mcmc(samples[:, usecols], labels=self._names[usecols], priors=priors,
                  ptrue=ptrue, nbins=nbins)


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
        return (a / (2.0 + b), 0.0, c, 2 * np.pi * np.exp(-log_P))


class FastRotationModeler(RotationModeler):
    _mu = (-17, -13, 0, 3.25)
    _sigma = (5.0, 5.7, 2.0, 0.7)
    _names = np.array(['$\ln{\sigma_n}$', '$\ln{B}$', '$\ln{C}$', '$\ln{L}$', '$\ln{P}$'])
    _bounds = [(-20, 0), (-20, 0), (-5, 5), (1.5, 5.0), (-0.69, 4.61)]

    def __init__(self, lc):
        super(FastRotationModeler, self).__init__(lc=lc)
        bounds = dict(log_B=(-20, 0), log_C=(-5, 5), log_L=(1.5, 5.0), log_P=(-0.69, 4.61))
        term = terms.JitterTerm(log_sigma=-17)
        term += CustomTerm(log_B=-13, log_C=0, log_L=3, log_P=2, bounds=bounds)
        self.gp = celerite.GP(term)

    def grad_nll(self, p, y):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(y)[1]


class StrongRotationModeler(RotationModeler):
    _mu = (-17, -13, 5.0, 1.9)
    _sigma = (5.0, 5.7, 1.2, 1.4)
    _names = np.array(['$\ln{\sigma_n}$', '$\ln{A}$', '$\ln{\ell}$', '$\ln{\Gamma}$', '$\ln{P}$'])
    _bounds = [(-20, 0), (-20, 0), (2, 8), (0, 3), (-0.69, 4.61)]

    def __init__(self, lc):
        super(StrongRotationModeler, self).__init__(lc=lc)
        kernel = kernels.ConstantKernel(-13, bounds=[(-20, 0)])
        kernel *= kernels.ExpSquaredKernel(150, metric_bounds=[(2, 8)])
        kernel *= kernels.ExpSine2Kernel(2, 2, bounds=[(0, 3), (-0.69, 4.61)])
        self.gp = george.GP(kernel, solver=george.HODLRSolver,
                            white_noise=-17, fit_white_noise=True)

    def grad_nll(self, p, y):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(y)

    def lnprior(self, p):
        gaussians = np.append([make_gauss(self._mu[i], self._sigma[i]) for i in
                               range(len(self._mu))], [self.prior])
        for i, (lo, hi) in enumerate(self._bounds):
            if p[i] > hi or p[i] < lo:
                return -np.inf
        if p[2] < p[-1]:
            return -np.inf
        lp = np.sum(np.log(gaussians[i](p[i])) for i in range(len(p)))
        return lp


def peaks(t, f, pmin=0.1):
    import matplotlib.pyplot as plt
    fs = 1 / np.median(np.diff(t))
    t -= min(t)
    ts = []
    hs = []
    qs = []
    for i in range(8):
        Pi = 2 ** i
        print('TESTANDO PI = {}'.format(Pi))
        if Pi >= max(t) / 2 or Pi <= pmin:
            continue
        y = filt(f, 1 / Pi, 1 / pmin, fs)
        ml = np.where(t >= 2 * Pi)[0][0]
        R = acf(y, maxlag=ml)
        if Pi >= 20:
            R = smooth(R, Box1DKernel(width=Pi // 10))
        plt.figure()
        plt.plot(t[:ml], R, 'k-')
        try:
            peaks, heights = find_peaks(R, t[:ml])
        except:
            plt.close()
            continue
        bp_acf = t[peaks][np.argmax(heights)]
        plt.axvline(bp_acf, color='r', ls='--', lw=1)
        ts.append(bp_acf)
        hs.append(max(heights))

        maxt = 20 * Pi / bp_acf

        def eps(p, x, y):
            mod = p[0] * np.exp(-x / p[1]) * np.cos(2 * np.pi * x / bp_acf)
            if p[1] > maxt:
                return np.ones_like(y)
            return np.square(y - mod)

        res = least_squares(eps, [1, 1], loss='soft_l1', f_scale=0.1, args=(t[:ml], R))
        print(res.x)
        A, tau = res.x
        R_model = A * np.exp(-t[:ml] / tau) * np.cos(2 * np.pi * t[:ml] / bp_acf)
        plt.plot(t[:ml], R_model, 'b-')
        plt.savefig('/home/alpaca/Desktop/TEST{}.png'.format(Pi))
        plt.close()
        eps = make_eps(ts[-1], 20 * Pi / ts[-1])
        ri = eps(res.x, t[:ml], R).sum()
        qs.append((tau / ts[-1]) * (ml * hs[-1] / ri))
        qs[-1] = max(qs[-1], 0)
    ts = np.array(ts)
    hs = np.array(hs)
    qs = np.array(qs)
    return ts, hs, qs


def make_model_acf(T):
    def model_acf(p, x):
        A = p[0]
        tau = p[1]
        return A * np.exp(-x / tau) * np.cos(2 * np.pi * x / T)

    return model_acf


def make_eps(T, maxt):
    def eps(p, x, y):
        mod = make_model_acf(T)
        if p[1] > maxt:
            return np.ones_like(y)
        return np.square(y - p[0] * np.exp(-x / p[1]) * np.cos(2 * np.pi * x / T))

    return eps


def make_prior(t, q):
    def prior(logP):
        tot = 0
        for ti, qi in zip(t, q):
            qi = max(qi, 0)
            gaussian1 = make_gauss(np.log(ti), 0.2)
            gaussian2 = make_gauss(np.log(ti / 2), 0.2)
            gaussian3 = make_gauss(np.log(2 * ti), 0.2)
            tot += qi * (0.9 * gaussian1(logP) +
                         0.05 * gaussian2(logP) +
                         0.05 * gaussian3(logP))
        tot /= q.sum()
        return tot

    return prior

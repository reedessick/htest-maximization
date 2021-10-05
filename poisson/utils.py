"""utilities for the poisson model of sparse x-ray data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from scipy.stats import poisson
from scipy.special import gammaln

#-------------------------------------------------

### general helper functions

def logfactorial(x):
    """natural logarithm of the factorial
    """
    return gammaln(x+1)

#-------------------------------------------------

### functions related to computing (and maximizing) the H-test statistic

def htest_significance(stat):
    """compute the approximate significance of the H-test statistic. Based on Eq. 5 of deJager+ (2010):
        https://ui.adsabs.harvard.edu/abs/2010A&A...517L...9D
    p(H>stat) = exp(-0.4*stat)
    """
    return np.exp(-0.4*stat)

def htest_statistic(data, mmin=1, mmax=20):
    """compute the H-test statistic for a set of observed phases. See Eqs 6, 7, and 11 of deJager+ (1989):
        https://ui.adsabs.harvard.edu/abs/1989A&A...221..180D
    return H-test statistic, corresponding number of harmonics
    """
    m = np.arange(mmin, mmax+1)

    # compute empirical trig moments
    coefs = np.array([empirical_trigonometric_moments(data, k) for k in m])

    # compute Z^2_m via a cumulative sum
    Z = 2*len(data) * np.cumsum(np.sum(coefs**2, axis=1))

    # compute H
    H = Z - 4*m + 4

    # maximize
    ind = np.argmax(H)

    # return
    return H[ind], m[ind]

def empirical_trigonometric_moments(data, k):
    """return the empirical trigonometric moments with harmonic k based on the phase data. See Eq 6 of deJager+ (1989):
        https://ui.adsabs.harvard.edu/abs/1989A&A...221..180D
    This is accomplished via Monte Carlo inner products between the data distribution and sinusoids.
    returns cosine_coefficient, sine_coefficient
    """
    return np.mean(np.cos(k*data)), np.mean(np.sin(k*data))

#-------------------------------------------------

### objects for generating synthetic data

class XrayProcess(object):
    """a model for generating realizations of periodic Poissonian X-ray events
    """

    def __init__(self, mean, *ac_components):

        ### store the input for interpolator construction
        self._mean = mean
        self._ac_components = ac_components
        self._phase_grid, self._phase_pdf, self._phase_cdf = self._init_interpolators(mean, *ac_components)

    @staticmethod
    def _init_interpolators(mean, *ac_components):
        ### construct an interpolation grid
        mmax = 0
        for m, a, d in ac_components: ### figure out the maximum harmonic present
            mmax = max(m, mmax)

        num_grid = 1 + 100*int(mmax+1) ### an approximate scaling of grid points
                                   ### make sure we sample the curve well enough
        phase_grid = np.linspace(0, 2*np.pi, num_grid)

        ### construct the pdf and cdf
        # instantiate the pdf and cdf using the DC component
        phase_pdf = np.ones(num_grid, dtype=float) * mean
        phase_cdf = mean * phase_grid

        for m, a, d in ac_components:
            assert m > 0, 'harmonic numbers must be positive (specify DC terms through "mean")'
            assert m%1==0, 'harmonic numbers must be integers!'
            phase_pdf += a * np.cos(m*phase_grid + d)
            phase_cdf += float(a)/m * (np.sin(m*phase_grid + d) - np.sin(d))

        assert np.all(phase_pdf >= 0), 'differential poisson rate must be positive semi-definite!'
        assert np.all(phase_cdf >= 0), 'phase CDF cannot be negative' ### probably redundant, but ok to check

        ### normalize interpolators
        phase_pdf /= phase_cdf[-1]
        phase_cdf /= phase_cdf[-1]

        # return
        return phase_grid, phase_pdf, phase_cdf

    @property
    def mean(self):
        return self._mean

    @property
    def mean_count(self):
        return self._mean * 2*np.pi ### the average number of counts, not the mean differential rate

    @property
    def ac_components(self):
        return self._ac_components

    @property
    def phase_grid(self):
        return self._phase_grid

    @property
    def phase_pdf(self):
        return self._phase_pdf

    @property
    def phase_cdf(self):
        return self._phase_cdf

    def binned_mean(self, low, high):
        """return the expected number of events with phases \in [low, high]
        """
        high = np.interp(high, self.phase_grid, self.phase_cdf)
        low = np.interp(low, self.phase_grid, self.phase_cdf)
        return self.mean * (high - low)

    def draw(self):
        """generate a realization of the Poisson process
        """
        num = poisson.rvs(self.mean_count) ### number of samples
        return self.rvs(size=num) ### distribute these through phase

    def rvs(self, size=1):
        """draw random variates distributed through phase via inverse transform sampling
        """
        return np.interp(np.random.random(size=size), self.phase_cdf, self.phase_grid)

    def __add__(self, other):
        """return a new instance that combines the Poissonian processes
        """
        assert isinstance(other, XrayProcess), 'can only add XrayProcess to another XrayProcess'
        return XrayProcess(self.mean+other.mean, *(self.ac_components+other.ac_components))

#------------------------

### object representing likelihood of observing binned phases

class Likelihood(object):
    """a model representing the Poisson likelihood for observations binned over phase
    """

    def __init__(self, process, bins):
        self.process = process
        if isinstance(bins, int): ### generate this number of equally spaced bins
            bins = np.linspace(0, 2*np.pi, bins+1)
        self.bins = bins
        self._check() ### sanity check the bins

    def _check(self):
        edge = 0.0
        for i in range(len(self.bins)-1):
            assert self.bins[i]==edge, 'bin %d does not start where previous bin ended!'
            edge = self.bins[i+1] ### move edge to the end of this bin
        assert edge==2*np.pi, 'bins do not end at 2*pi'

    @property
    def nbins(self):
        return len(self.bins)

    @property
    def mean(self):
        return self.process.mean

    @property
    def binned_means(self):
        return np.array([self.process.binned_mean(bins[i], bins[i+1]) for i in range(self.nbins)])

    def bin(self, data):
        """map data (an array of phases) into counts in each bin
        """
        return np.histogram(data, self.bins)[0] ### only return the array of counts

    def prob(self, binned_data):
        return np.exp(self.logprob(binned_data))

    def logprob(self, binned_data):
        """the natural log of the probability of observing binned_data given our binning and our process
        """
        return -self.mean + np.sum(binned_data*np.log(self.binned_means) - logfactorial(binned_data))

"""utilities for the poisson model of sparse x-ray data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from scipy.special import gammaln

#-------------------------------------------------

def logfactorial(x):
    """natural logarithm of the factorial
    """
    return gammaln(x+1)

#-------------------------------------------------

class XrayProcess(object):
    """a model for generating realizations of periodic Poissonian X-ray events
    """

    def __init__(self, mean, *ac_components):
        self.mean = mean
        self.ac_components = list(ac_components)
        self._check() ### check that the differential rate is always positive

    def _check(self):
        raise NotImplementedError('''\
sanity check the signal model to make sure the differential rate is always non-negative
do this via brute-force by generating sampling that is much finer than the highest harmonic present and directly checking the model everywhere?
''')

    def binned_mean(self, low, high):
        """return the expected number of events in phases \in [low, high]
        """
        raise NotImplementedError('integrate this analytically. Or numerically using our "really fine" grid?')

    def rvs(size=1):
        """draw random variates from this model
        """
        raise NotImplementedError('draw samples from this analytically? Or compute CDF with finer spacing than the highest harmonic and then simply inverse-transform sample?')

    def __add__(self, other):
        """return a new instance that combines the Poissonian processes
        """
        assert isinstance(other, XrayProcess), 'can only add XrayProcess to another XrayProcess'
        return XrayProcess(self.mean+other.mean, *(self.ac_components+other.ac_components))

#------------------------

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

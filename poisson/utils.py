"""utilities for the poisson model of sparse x-ray data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

class XrayProcess(object):
    """a model for generating realizations of periodic Poissonian X-ray events
    """

    def __init__(self, mean, *fourier_components):
        self.mean = mean
        self.fourier_components = []
        raise NotImplementedError('''\
if fourier_components is empty, set this up as independent of phase (m=0 harmonic only)
store the fourier components
normalize them so they produce the overall mean
sanity check the signal model to make sure the differential rate is always non-negative
''')

    def rvs(size=1):
        """draw random variates from this model
        """
        raise NotImplementedError

    def __add__(self, other):
        """return a new instance that combines the Poissonian processes
        """
        assert isinstance(other, XrayProcess), 'can only add XrayProcess to another XrayProcess'
        raise NotImplementedError

    def binned_mean(self, low, high):
        """return the expected number of events in phases \in [low, high]
        """
        raise NotImplementedError

#------------------------

class Likelihood(object):
    """a model representing the Poisson likelihood for observations binned over phase
    """

    def __init__(self, process, bins):
        self.process = process
        if isinstance(bins, int): ### generate this number of equally spaced bins
            bins = np.linspace(0, 2*np.pi, bins+1)
        self.bins = bins

    def bin(self, data):
        raise NotImplementedError('map the data into counts per bin')

    def prob(self, binned_data):
        raise NotImplementedError('compute the probability over all bins together')

    def logprob(self, binned_data):
        raise NotImplementedError('compute the natural log of the probability over all bins together')

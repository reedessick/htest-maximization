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

class BinnedLikelihood(object):
    """a model representing the Poisson likelihood for observations binned over phase
    """

    def __init__(self, bins, process=None):
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
        return len(self.bins) - 1

    @property
    def mean(self):
        if self.process is None:
            raise ValueError('must set a process!')
        return self.process.mean_count

    @property
    def binned_means(self):
        if self.process is None:
            raise ValueError('must set a process!')
        return np.array([self.process.binned_mean(self.bins[i], self.bins[i+1]) for i in range(self.nbins)])

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

    def mle(self, binned_data, harmonics=[]):
        """compute the maximum likelihood estiamte
        """
        # first compute an estimate using an approximate likelihood
        lambda0, ac_components = self.approximate_mle(binned_data, harmonics=harmonics)
        raise NotImplementedError('''\
take the linearized estimate and use it as the seed for a numeric maximization of logprob
''')

    def approximate_mle(self, binned_data, harmonics=[]):
        """compute the maximum likelihood estimate for the Poisson process with M harmonics based on the binned data
        We solve for the parameters of a model described by:
            lambda(phi) = lambda0 * ( 1 + sum_{m=1}^M ( a_m * cos(m*phi) + b_m * sin(m*phi) ) )
        under the assumption that
            a_m, b_m << 1 for all m
        so that we can expand the likelihood as a quadratic function of (a_m, b_m)
        """
        ### first compute the mean rate
        lambda0 = np.sum(binned_data) / (2*np.pi) ### this one is easy

        ### construct a set of linear equations for the fourier coefficients

        # the harmonics we include in the signal model
        M = len(harmonics)
        harmonics = np.array(harmonics)
        inv_harmonics = 1./harmonics

        # holders for the matrix equation: coeffs = inv(matrix) * const
        const = np.empty((2*M, 1), dtype=float)
        matrix = np.empty((2*M, 2*M), dtype=float)

        # set up convenient measures for bin edges
        left = self.bins[:-1] ### lower bin edges
        right = self.bins[1:] ### upper bin edges
        dphi = right - left   ### size of each bin
        dphi2 = dphi**2

        # compute trig functions once
        dsin = np.empty((M, self.nbins), dtype=float)
        dcos = np.empty((M, self.nbins), dtype=float)
        for ind, m in enumerate(harmonics):
            dsin[ind, :] = np.sin(m*right) - np.sin(m*left)
            dcos[ind, :] = np.cos(m*right) - np.cos(m*left)

        # iterate over harmonics, filling in matrix elements
        for ind, m in enumerate(harmonics):
            mind = 2*ind

            # fill in the constant vector
            const[mind, 0] = np.sum(binned_data * dsin[ind] / dphi)   # for a_m
            const[mind+1, 0] = np.sum(binned_data * dcos[ind] / dphi) # for b_m

            # fill in this row of the matrix
            for jnd, n in enumerate(harmonics):
                nind = 2*jnd

                # fill in row for a_m --> two columns corresponding to a_n and b_n
                matrix[mind, nind] = inv_harmonics[jnd] * np.sum(binned_data / dphi2 * dsin[ind] * dsin[jnd])
                matrix[mind, nind+1] = - inv_harmonics[jnd] * np.sum(binned_data / dphi2 * dsin[ind] * dcos[jnd])

                # fill in row for b_m --> two columns corresponding to a_n and b_n
                matrix[mind+1, nind] = inv_harmonics[jnd] * np.sum(binned_data / dphi2 * dcos[ind] * dsin[jnd])
                matrix[mind+1, nind+1] = - inv_harmonics[jnd] * np.sum(binned_data / dphi2 * dcos[ind] * dcos[jnd])

        # solve for the coefficients
        coef = np.dot(np.linalg.inv(matrix), const)

        # now, convert coefficients into "ac_component" parametrization:
        #     lambda = lambda0 + sum_m ( a_m * cos(m*phi + d_m) )
        coef = lambda0 * coef.reshape((M, 2))

        a_m = np.sum(coef**2, axis=1)**0.5 ### amplitudes
        d_m = np.arctan2(-coef[:,1]/a_m, coef[:,0]/a_m)

        ### return parameters of MLE process
        return lambda0, list(zip(harmonics, a_m, d_m))

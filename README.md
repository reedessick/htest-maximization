# htest-maximization

A few simple toy models to examine the possible impact of selecting data segements based on the maximization of a detection statistic.

In particular, we examine the procedure detailed in 

  * Guillot et al, *NICER X-Ray Observations of Seven Nearby Rotation-powered Millisecond Pulsars*, ApJ Let 887:L27 (2019).

in which data segments are ordered by increasing count rate and cumulatively included until a detection statistic is maximized (Sec 3.2). The detection statistic is the H-test

  * de Jager et al, *A powerful test for weak periodic signals with unkown light curve shape in sparse data*, A&A 221 (1989)

which searches for deviations for uniformity in phase-folded histograms by maximizing a measure of the deviation from uniformity over model light curves with different numbers of harmonics. This procedure may select a subset of data that de facto biases the inferred modulation depth to larger values (larger variability in the light curve). If this is the case, it could impact the subsequent estimate of the compactnes (and therefore the mass and/or radius) of pulsars observed by NICER.

---

## Toy models

### univariate-exponential

We investigate a simple toy model for the minimum value in a shifted exponential distribution.
The presence of signals is denoted by a nonzero shift, and we consider including data incrementally from the largest to the smallest observed values.

We note that the H-test statistic is expected to be exponentially distributed.

### univariate-gaussian

We investigate a simple toy model of the mean of a Gaussian distribution.
The presence of a signal is denoted by a mean that is greater than zero, and we consider including data incrementally from the largest to the smallest observed values.

### poisson

We investigate a toy model in which the observed data is generated as the sum of

  - background: a stationary poisson process that does not depend on the rotation phase
  - foreground: a modulated poisson process for which the event rate depends on the rotation phase: `dlambda/dphi != 0`

Our toy model generates a set of i.i.d. data segments with sparse counts from this model and then applies the data selection procedure described in Guillot+(2019).
The severity of biases introduced are measured with simple estimates of the posterior for the modulated signal light curve parameters (rather than the compactness, mass, and radius of a star).

---

**TO DO**

  * [ ] implement basic model for sinusoidal light curve
  * [ ] implement htest statistic
  * [ ] perform many trials of data selection and see if there is an effect
    - quantify the size, particularly on the relative depth of the lightcurve

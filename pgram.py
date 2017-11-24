from pylab import *

def pgram(ts, ys, dys, P):
    r"""Returns the estimated mean and sine- and cosine-term coefficients
    for the time series.

    :param ts: An array of times.

    :param ys: An array of values.

    :param dys: An array of uncertainties on the values.

    :param P: The period of the sinusoidal components.

    :return: ``(mu, A, B)``, where ``y ~ mu + A*cos(2*pi*t/P) +
      B*sin(2*pi*t/P)``

    The coefficients are determined by minimising the squared
    residual, weighted by the observational uncertaintes.  This is the
    maximum likelihood estimate for ``mu``, ``A``, and ``B`` if the
    data, ``ys`` are Gaussian-distributed with uncertainty ``dys``:

    ..math::

      \min_{\mu, A, B} \sum_{i=1}^N \left(\frac{ y_i - \mu - A \cos\left( \frac{2 \pi t_i}{P} \right) - B \sin \left( \frac{2 \pi t_i}{P} \right)}{\delta y_i} \right)^2

    (This method is sometimes called "weighted least squares.")

    """

    # It is easiest to do this via standard linear algebra operations;
    # first, we form the "response matrix," M, so that, given
    # parameters p = (mu, A, B), we have y = M*p.
    cos_terms = cos(2.0*pi*ts/P)
    sin_terms = sin(2.0*pi*ts/P)
    const_terms = ones(ts.shape[0])

    M = column_stack((const_terms, cos_terms, sin_terms))

    # Now we want to solve, in a least-squares sense, the matrix
    # equation (M*p)/dys = y/dys for p (dividing by dys makes the
    # method *weighted* least squares).  We need to reshape the dys
    # vector so that the *rows* of M are divided by the corresponding
    # dys.

    p, _, _, _ = np.linalg.lstsq(M / reshape(dys, (-1, 1)), ys/dys)

    return p

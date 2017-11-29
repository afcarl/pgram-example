from pylab import *

def pgram(ts, ys, dys, P, nharmonics=1):
    r"""Returns the estimated mean and sine- and cosine-term coefficients
    for the time series.

    :param ts: An array of times.

    :param ys: An array of values.

    :param dys: An array of uncertainties on the values.

    :param P: The period of the sinusoidal components.

    :param nharmonics: The number of harmonics of the period, ``P`` to
      consider.

    :return: ``((mu, A, B), rss)``, where ``y ~ mu + A*cos(2*pi*t/P) +
      B*sin(2*pi*t/P)`` (or equivalent for the multiple-harmonic
      return), and ``rss`` is the residuals squared and summed.

    The coefficients are determined by minimising the squared
    residual, weighted by the observational uncertaintes.  This is the
    maximum likelihood estimate for ``mu``, ``A``, and ``B`` if the
    data, ``ys`` are Gaussian-distributed with uncertainty ``dys``:

    ..math::

      \min_{\mu, A, B} \sum_{i=1}^N \left(\frac{ y_i - \mu - A \cos\left( \frac{2 \pi t_i}{P} \right) - B \sin \left( \frac{2 \pi t_i}{P} \right)}{\delta y_i} \right)^2

    (This method is sometimes called "weighted least squares.")

    If ``nharmonics > 1``, then the coefficients will be returned in
    the order ``[A1, A2, A3, ..., B1, B2, B3, ...]`` where ``A1`` is
    the coefficient in front of the fundamental cosine term, ``A2`` in
    front of the first harmonic cosine term, etc, and ``B1``, ``B2``,
    etc for the sine terms.

    """

    # It is easiest to do this via standard linear algebra operations;
    # first, we form the "response matrix," M, so that, given
    # parameters p = (mu, A, B), we have y = M*p.
    if nharmonics == 1:
        cos_terms = cos(2.0*pi*ts/P)
        sin_terms = sin(2.0*pi*ts/P)
    else:
        cos_terms = []
        sin_terms = []
        for i in range(nharmonics):
            j = i + 1
            cos_terms.append(cos(2.0*pi*j*ts/P))
            sin_terms.append(sin(2.0*pi*j*ts/P))
        cos_terms = transpose(array(cos_terms))
        sin_terms = transpose(array(sin_terms))
        
    const_terms = ones(ts.shape[0])

    M = column_stack((const_terms, cos_terms, sin_terms))

    # Now we want to solve, in a least-squares sense, the matrix
    # equation (M*p)/dys = y/dys for p (dividing by dys makes the
    # method *weighted* least squares).  We need to reshape the dys
    # vector so that the *rows* of M are divided by the corresponding
    # dys.

    p, rss, _, _ = np.linalg.lstsq(M / reshape(dys, (-1, 1)), ys/dys)

    return p, rss

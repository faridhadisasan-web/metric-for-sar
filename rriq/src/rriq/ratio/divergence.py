import numpy as np
from rriq.ratio.distributions import fit_gamma


def kl_divergence_gamma(ratio_data: np.ndarray, looks: int = 1) -> float:
    """
    Approximates the Kullback-Leibler divergence between the empirical ratio
    distribution and a theoretical Gamma distribution with shape=looks and scale=1/looks.
    Lower values indicate better speckle preservation (closer to theoretical noise model).
    """
    shape_empirical, _, scale_empirical = fit_gamma(ratio_data)
    # This is a simplified analytic KL divergence for two Gamma distributions
    # KL(Gamma(k1, theta1) || Gamma(k2, theta2))

    k1, theta1 = shape_empirical, scale_empirical
    k2, theta2 = looks, 1.0 / looks

    from scipy.special import digamma, gammaln

    kl = (
        (k1 - k2) * digamma(k1)
        - gammaln(k1)
        + gammaln(k2)
        + k2 * (np.log(theta2) - np.log(theta1))
        + k1 * (theta1 - theta2) / theta2
    )

    return float(kl)

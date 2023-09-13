"""Functions for GOPH 419 Lab Assignment #1."""

import numpy as np


def arcsin(x):
    """Compute the inverse sine of x on the range [-pi/2, pi/2].

    Parameters
    ----------
    x : float
        Argument of inverse sine function.
        Assumed to be on the domain [-1.0, +1.0].

    Returns
    -------
    float
        Inverse sine of x.

    Raises
    ------
    ValueError
        If abs(x) > 1.0.

    Notes
    -----
    The algorithm implemented is based on a series solution
    for (arcsin(x))**2 presented in:
        Borwein, J.M. and Chamberland, M. (2007). Integer Powers of Arcsin,
            Int. J. Math. Math. Sci., Art. 19381, doi: 10.1155/2007/19381.
    For negative values of x, the sign is stored,
    the result is computed from the series based on the absolute value,
    and the sign is restored in the return since arcsin is an odd function.
    Convergence may be slow for x values close to 1.0.
    """
    sign = 1.0
    if x < 0.0:
        sign = -1.0     # store the sign for return later
        x = np.abs(x)   # algorithm is based on absolute value
    if x > 1.0:
        raise ValueError(f"input abs({sign * x}) > 1.0 is out of range")
    eps_s = 0.5e-5  # stopping criterion, so that at least 5 sig figs converge
    if x < eps_s:
        return sign * x  # return early for small x
    two_x = 2.0 * x
    n = 0
    max_n = 100     # maximum number of iterations, to stop slow convergence
    fact_n = 1
    fact_2n = 1     # for storing factorials, avoid iterative factorial call
    result = 0.0
    eps_a = 1.0     # initialize approximate relative error
    while eps_a > eps_s and n < max_n:
        n += 1
        two_n = 2 * n
        fact_n *= n
        fact_2n *= two_n * (two_n - 1)
        term = 0.5 * (two_x ** two_n) / (n**2 * fact_2n / fact_n**2)
        result += term
        eps_a = term / result
    return sign * np.sqrt(result)


def launch_angle(ve_v0, alpha):
    """Calculate the launch angle from vertical given
    velocity ratio and target altitude ratio.

    Parameters
    ----------
    ve_v0 : float
        Ratio of escape velocity to terminal velocity.
    alpha : float
        Target altitude ratio relative to Earth's radius.

    Returns
    -------
    float
        Launch angle from vertical in radians.

    Raises
    ------
    ValueError
        If the combination of ve_v0 and alpha is not valid.

    Notes
    -----
    The calculation is based on the formula:
        sin(phi0) = (1 + alpha) * sqrt(1 - alpha / (1 + alpha) * ve_v0**2)
    where phi0 is the launch angle. This equation comes from conservation of
    kinetic and gravitational potential energy and angular momentum balance.
    A meaningful result can only be obtained when:
        ve_v0 < 1.0
    and
        alpha <= 1 / (ve_v0**2 - 1)
    and a ValueError is raised if that is not the case.
    """
    if ve_v0 < 1.0:
        raise ValueError(f"invalid value ve_v0 = {ve_v0} > 1.0")
    d = 1.0 - alpha / (1.0 + alpha) * ve_v0**2
    if d < 0.0:
        alpha_max = 1.0 / (ve_v0**2 - 1.0)
        raise ValueError(f"invalid value alpha = {alpha} > {alpha_max}")
    x = (1.0 + alpha) * np.sqrt(d)
    return arcsin(x)


def launch_angle_range(ve_v0, alpha, tol_alpha):
    """Calculate the range of launch angles for a given
    velocity ratio, target altitude ratio, and tolerance.

    Parameters
    ----------
    ve_v0 : float
        Ratio of escape velocity to terminal velocity.
    alpha : float
        Target altitude ratio relative to Earth's radius.
    tol_alpha : float
        Tolerance range for maximum altitude.

    Returns
    -------
    numpy.array, shape = (,2)
        Vector of minimum and maximum launch angles in radians.

    Notes
    -----
    This function is a wrapper that attempts to call launch_angle()
    with inputs ve_v0 and:
        (1 + tol_alpha) * alpha
        (1 - tol_alpha) * alpha
    No error check is performed and launch_angle() may raise a ValueError
    if the combination is invalid.
    """
    alpha_max = (1.0 + tol_alpha) * alpha
    alpha_min = (1.0 - tol_alpha) * alpha
    return np.array([launch_angle(ve_v0, alpha_max),
                     launch_angle(ve_v0, alpha_min)])


def min_altitude_ratio(ve_v0):
    """Utility function for computing minimum possible peak altitude ratio
    for a given velocity ratio.

    Parameters
    ----------
    ve_v0 : float
        Ratio of escape velocity to terminal velocity.

    Returns
    -------
    float
        Minimum possible peak altitude ratio.

    Notes
    -----
    This range results from the formula for launch angle:
        sin(phi0) = (1 + alpha) * sqrt(1 - alpha / (1 + alpha) * ve_v0**2)
    The right-hand-side of the launch angle formula must be <= 1.0,
    since the maximum possible launch angle is pi/2.
    For this condition, we can obtain the limit:
        alpha <= -(ve_v0**2 - 2) / (ve_v0**2 - 1)
    This limit is irrelevant for large ve_v0 (small initial velocity),
    specifically when ve_v0 > sqrt(2) the limit is negative,
    so in that case there is a trivial minimum altitude ratio of 0.0.
    """
    if ve_v0 <= 1.0:
        raise ValueError(f"invalid velocity ratio: {ve_v0}, must be > 1.0")
    return np.max([0.0, -(ve_v0**2 - 2.0) / (ve_v0**2 - 1.0)])


def max_altitude_ratio(ve_v0):
    """Utility function for computing maximum possible peak altitude ratio
    for a given velocity ratio.

    Parameters
    ----------
    ve_v0 : float
        Ratio of escape velocity to terminal velocity.

    Returns
    -------
    float
        Maximum possible peak altitude ratio.

    Notes
    -----
    This range results from the formula for launch angle:
        sin(phi0) = (1 + alpha) * sqrt(1 - alpha / (1 + alpha) * ve_v0**2)
    The part under the square root must be >= 0.0
    to obtain a real-valued result. Since, for a given ve_v0,
    increasing alpha tends to decrease the value under the square root,
    we can use this to obtain a maximum possible alpha
    (i.e. maximum possible peak altitude)
    that can be reached for given initial velocity:
        alpha <= 1 / (ve_v0**2 - 1)
    As ve_v0 becomes large (i.e. initial velocity becomes small),
    this limit approaches 0.0, as one would expect.
    """
    if ve_v0 <= 1.0:
        raise ValueError(f"invalid velocity ratio: {ve_v0}, must be > 1.0")
    return 1.0 / (ve_v0**2 - 1.0)


def min_velocity_ratio(alpha):
    """Utility function for computing minimum possible velocity ratio
    for a given target peak altitude ratio.

    Parameters
    ----------
    alpha : float
        Target peak altitude ratio.

    Returns
    -------
    float
        Minimum possible velocity ratio ve_v0.

    Notes
    -----
    This range results from the formula for launch angle:
        sin(phi0) = (1 + alpha) * sqrt(1 - alpha / (1 + alpha) * ve_v0**2)
    The right-hand-side of the launch angle formula must be <= 1.0,
    since the maximum possible launch angle is pi/2.
    This corresponds to the minimum possible ve_v0
    (i.e. maximum possible launch velocity)
    to reach a given peak altitude when all energy is going into
    angular (orbital) momentum:
        ve_v0 >= sqrt((2 + alpha) / (1 + alpha))
    As alpha becomes large, this limit approaches 1.0,
    which implies an initial velocity equal to the escape velocity.
    """
    if alpha <= 0.0:
        raise ValueError(f"invalid altitude ratio: {alpha}, must be > 0.0")
    return np.sqrt((2.0 + alpha) / (1.0 + alpha))


def max_velocity_ratio(alpha):
    """Utility function for computing maximum possible velocity ratio
    for a given target peak altitude ratio.

    Parameters
    ----------
    alpha : float
        Target peak altitude ratio.

    Returns
    -------
    float
        Maximum possible velocity ratio ve_v0.

    Notes
    -----
    This range results from the formula for launch angle:
        sin(phi0) = (1 + alpha) * sqrt(1 - alpha / (1 + alpha) * ve_v0**2)
    The part under the square root must be >= 0.0
    to obtain a real-valued result. Since, for a given alpha,
    increasing ve_v0 tends to decrease the value under the square root,
    we can use this to obtain a maximum possible ve_v0
    (i.e. minimum possible launch velocity)
    to reach a given peak altitude:
        ve_v0 <= sqrt((1 + alpha) / alpha)
    As alpha becomes large, this limit approaches 1.0,
    which implies an initial velocity equal to the escape velocity.
    """
    if alpha <= 0.0:
        raise ValueError(f"invalid altitude ratio: {alpha}, must be > 0.0")
    return np.sqrt((1.0 + alpha) / alpha)

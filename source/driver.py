"""Driver script for Assignment #1.

See Also
--------
- Assignment description for mathematical and physics background.
- launch.py for documentation of variables and functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from launch import (
        min_altitude_ratio,
        max_altitude_ratio,
        min_velocity_ratio,
        max_velocity_ratio,
        launch_angle_range,
)


def main():
    """Plot launch angle ranges given
    values for ve_v0, alpha, and tol_alpha.
    """

    # plot launch angle range for a fixed velocity ratio
    ve_v0 = 2.0
    tol_alpha = 0.04
    alpha_min = min_altitude_ratio(ve_v0) / (1.0 - tol_alpha)
    alpha_max = max_altitude_ratio(ve_v0) / (1.0 + tol_alpha)
    alpha_range = np.linspace(alpha_min, alpha_max, 20)

    phi0_min = np.zeros_like(alpha_range)
    phi0_max = np.zeros_like(alpha_range)

    for k, alpha in enumerate(alpha_range):
        phi0 = launch_angle_range(ve_v0, alpha, tol_alpha)
        phi0_min[k] = phi0[0]
        phi0_max[k] = phi0[1]

    plt.figure()
    plt.plot(alpha_range, phi0_min, '-k')
    plt.plot(alpha_range, phi0_max, '--k')
    plt.title(r"$\frac{v_e}{v_0} = 2.0$, $tol_{\alpha} = 0.04$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\phi_0$")
    plt.legend([r"$(\phi_0)_{min}$", r"$(\phi_0)_{max}$"])
    plt.savefig("../figures/ve_v0_2.png")

    # plot launch angle range for a fixed target altitude
    alpha = 0.25
    tol_alpha = 0.04
    ve_v0_min = min_velocity_ratio((1.0 - tol_alpha) * alpha)
    ve_v0_max = max_velocity_ratio((1.0 + tol_alpha) * alpha)
    ve_v0_range = np.linspace(ve_v0_min, ve_v0_max, 20)

    phi0_min = np.zeros_like(ve_v0_range)
    phi0_max = np.zeros_like(ve_v0_range)

    for k, ve_v0 in enumerate(ve_v0_range):
        phi0 = launch_angle_range(ve_v0, alpha, tol_alpha)
        phi0_min[k] = phi0[0]
        phi0_max[k] = phi0[1]

    plt.figure()
    plt.plot(ve_v0_range, phi0_min, '-k')
    plt.plot(ve_v0_range, phi0_max, '--k')
    plt.title(r"$\alpha = 0.25$, $tol_{\alpha} = 0.04$")
    plt.xlabel(r"$\frac{v_e}{v_0}$")
    plt.ylabel(r"$\phi_0$")
    plt.legend([r"$(\phi_0)_{min}$", r"$(\phi_0)_{max}$"])
    plt.savefig("../figures/alpha_0p25.png")


if __name__ == "__main__":
    main()

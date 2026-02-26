"""
Auxiliary functions to generate mock data
=========================================

This module implements some auxiliary functions to create mock datasets for Skykatana

* make_random_sphere() > Generate N random (ra, dec) points uniformly over the shpere
* generate_powerlaw_radii() > Generate power-radii between a minimum and a maximum radius
* make_random_stars() > use functions above to generate a (ra,dec,radius) mock dataset

"""


import numpy as np
import pandas as pd


def make_random_sphere(N: int, seed: int = 0, ralim=None, declim=None):
    """
    Generate N random (ra, dec) points uniformly on the sphere, optionally
    restricted to a RA/Dec rectangle.

    Parameters
    ----------
    N : int
        Number of points.
    seed : int
        RNG seed.
    ralim : sequence of 2 floats, optional
        Right ascention boundaries in degrees. If None then stars are within [0,360]deg
    declim : sequence of 2 floats, optional
        Declination boundaries in degrees. If None then stars are within [-90,90]deg

    Returns
    -------
    ra, dec : np.ndarray
        Arrays of shape (N,) in degrees. ra in [0, 360), dec in [-90, 90].
    """
    rng = np.random.default_rng(seed)

    # RA sampling (uniform in angle)
    if ralim is None:
        ra = rng.uniform(0.0, 360.0, N)
    else:
        ra1, ra2 = float(ralim[0]), float(ralim[1])
        ra1 = ra1 % 360.0
        ra2 = ra2 % 360.0

        # If ra2 >= ra1: simple interval [ra1, ra2]
        # If ra2 <  ra1: wrap interval [ra1, 360) U [0, ra2]
        span = (ra2 - ra1) % 360.0  # in [0, 360)
        if span == 0.0:
            # Ambiguous: could mean full circle or empty. Here we treat as full circle.
            ra = rng.uniform(0.0, 360.0, N)
        else:
            ra = (ra1 + rng.uniform(0.0, span, N)) % 360.0

    # Dec sampling (uniform in sin(dec))
    if declim is None:
        u = rng.uniform(-1.0, 1.0, N)
    else:
        dec1, dec2 = float(declim[0]), float(declim[1])
        # allow to pass in any order
        dmin, dmax = (dec1, dec2) if dec1 <= dec2 else (dec2, dec1)

        # clip to physical range
        dmin = max(-90.0, dmin)
        dmax = min( 90.0, dmax)
        if dmax < dmin:
            raise ValueError("declim does not overlap [-90, 90] after clipping.")

        umin = np.sin(np.deg2rad(dmin))
        umax = np.sin(np.deg2rad(dmax))
        u = rng.uniform(umin, umax, N)

    dec = np.rad2deg(np.arcsin(u))

    return ra.astype(np.float64), dec.astype(np.float64)

    

def generate_powerlaw_radii(N: int = 10000, min_radius: float = 20.0, max_radius: float = 200.0,
    power_law_index: float = -2.2, seed: int | None = None, arcsec_flag: bool = False):
    """
    Draw radii r in [min_radius, max_radius] (arcsec) from a differential power-law:
        p(r) ∝ r^{power_law_index}.

    Parameters
    ----------
    N : int
        Number of samples.
    min_radius, max_radius : float
        Minimum/maximum radius (arcsec). Must satisfy 0 < min_radius < max_radius.
    power_law_index : float
        Exponent of the differential distribution (gamma).
        Example: gamma = -1.6 means p(r) ∝ r^{-1.6}.
    arcsec_flag : bool
        If True, return radius in arcsec. If False return radius in degrees (default)
    seed : int | None
        RNG seed.

    Returns
    -------
    radii : np.ndarray
        Radii in arcsec, shape (N,), dtype float64.
    """
    
    rmin, rmax = float(min_radius), float(max_radius)
    
    if not (rmin > 0.0 and rmax > 0.0):
        raise ValueError("min_radius and max_radius must be > 0.")
    if not (rmax > rmin):
        raise ValueError("max_radius must be greater than min_radius.")

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, N)

    gamma = float(power_law_index)
    alpha = gamma + 1.0  # exponent in the CDF

    if np.isclose(alpha, 0.0):
        # gamma = -1 -> p(r) ∝ 1/r, CDF is logarithmic
        radii = rmin * (rmax / rmin) ** u
    else:
        radii = ((rmax**alpha - rmin**alpha) * u + rmin**alpha) ** (1.0 / alpha)

    if not(arcsec_flag): radii = radii/3600.
        
    return radii.astype(np.float64)



def make_random_stars(N: int, ralim=None, declim=None,
    min_radius: float = 20.0, max_radius: float = 200.0,
    power_law_index: float = -2.2, arcsec_flag :bool = False, seed: int = 0):
    
    """
    Generate random stars with (ra, dec) uniformly on the sphere (optionally in a
    RA/Dec rectangle) and radii drawn from a power-law in [min_radius, max_radius].

    Ra-Dec-radii are in degrees unless arcsec_flag=True, in which case radii will
    be returned in arcsec.

    Parameters
    ----------
    N : int
        Number of stars
    ralim : sequence of 2 floats, optional
        Right ascention boundaries in degrees. If None then stars are within [0,360]deg
    declim : sequence of 2 floats, optional
        Declination boundaries in degrees. If None then stars are within [-90,90]deg
    min_radius, max_radius : float
        Minimum/maximum radius (arcsec)
    power_law_index : float
        Exponent of the differential distribution (gamma)
        Example: gamma = -1.6 means p(r) ∝ r^{-1.6}.
    arcsec_flag : bool
        If True, return radius in arcsec. If False (default), return radius in degrees 
    seed : int | None
        RNG seed

    Returns
    -------
    df : pandas dataframe
        Dataframe with columns 'ra', 'dec', 'radius'
    """
    # Use independent RNGs for positions vs sizes, but reproducible from one seed
    rng = np.random.default_rng(seed)
    seed_pos, seed_size = rng.integers(0, 2**63 - 1, size=2, dtype=np.int64)

    # Generate positions
    ra, dec = make_random_sphere(N, seed=int(seed_pos), ralim=ralim, declim=declim)
    # Generate radii
    radii = generate_powerlaw_radii( N=N, min_radius=min_radius, max_radius=max_radius,
        power_law_index=power_law_index, seed=int(seed_size), arcsec_flag=arcsec_flag)
    # Create output dataframe
    df = pd.DataFrame()
    df['ra'] = ra
    df['dec'] = dec
    df['radius'] = radii
    return df
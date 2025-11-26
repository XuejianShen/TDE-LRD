"""
Tidal Disruption Event (TDE) rate calculation using Hernquist density profile
with the Stone & Metzger (2016) loss cone formalism.

This module implements the calculation of TDE rates for stars orbiting a 
supermassive black hole embedded in a Hernquist stellar density profile.
The method follows the angular momentum diffusion approach in Stone & Metzger (2016).

Unit convention:
- Length: pc
- Velocity: km/s
- Mass: Msun
- Time: pc/(km/s) (internal units)
  Note: To convert to years, multiply by pc_per_kms_to_yr from header.py
- Potential (psi): (km/s)^2
- Angular momentum (J): pc * km/s
- Energy (E): (km/s)^2
"""

import numpy as np
import astropy.constants as con
from scipy.integrate import cumulative_trapezoid
from header import *
import density_profile as profile
import eddington_inversion as edd

# unit convention: pc, km/s, Msun, yr

def make_monotonic(x, y):
    # sort and remove duplicates
    idx = np.argsort(x)
    xs = x[idx]; ys = y[idx]
    mask = np.concatenate(([True], np.diff(xs) > 0))
    return xs[mask], ys[mask] 

def compute_Jc_P_E(E_grid, r_grid, psi_r, vc2_arr):
    """
    Compute circular angular momentum Jc(E), orbital period P(E), and apocenter r_a(E).
    
    For each binding energy E, this function computes:
    - Jc(E): Angular momentum of a circular orbit at that energy
    - P(E): Orbital period for a radial orbit (eccentricity e=1)
    - r_a(E): Apocenter radius for a radial orbit (where psi(r_a) = E)
    - r_c(E): Radius of the circular orbit
    
    Parameters
    ----------
    E_grid : array_like
        Grid of binding energies (E = -1/2*v^2 + psi) in (km/s)^2
    r_grid : array_like
        Radial grid in pc
    psi_r : array_like
        Potential at r_grid
    vc2_arr : array_like
        Circular velocity squared at r_grid
    
    Returns
    -------
    r_a : array
        Apocenter radius for each energy in pc
    Period : array
        Orbital period for each energy in pc/(km/s)
    Jc : array
        Circular angular momentum for each energy in pc*km/s
    r_c : array
        Circular orbit radius for each energy in pc
    """
    # Binding energy of a circular orbit at each radius
    # eps_c = psi - 0.5*v_circ^2 (binding energy = -orbital energy)
    eps_c_r = psi_r - 0.5 * vc2_arr
    
    # Invert the mapping r -> eps_c to get r_c(E)
    # Reverse arrays since potential decreases with radius (eps increases)
    eps_s, r_s = make_monotonic(eps_c_r[::-1], r_grid[::-1])
    r_c = np.interp(np.clip(E_grid, eps_s[0], eps_s[-1]), eps_s, r_s)
    # Get circular velocity at r_c
    vc2_c = np.interp(r_c, r_grid, vc2_arr)
    # Circular angular momentum: Jc = r * v_circ
    Jc = r_c * np.sqrt(vc2_c)
    
    # Invert the mapping r -> psi to get apocenter r_a(E) for highly eccentric orbits
    # For a radial orbit (e=1), the apocenter is where psi(r_a) = E
    psi_s, r_for_psi = make_monotonic(psi_r[::-1], r_grid[::-1])
    r_a = np.interp(np.clip(E_grid, psi_s[0], psi_s[-1]), psi_s, r_for_psi)
    
    # Compute orbital period P(E) using the radial action integral
    # P = 2 * integral[dr / v_r] where v_r = sqrt(2*(psi - E))
    # Units: dr [pc] / v_r [km/s] = pc/(km/s), so P [pc/(km/s)]
    period_factor = 2.0
    Period = np.full_like(E_grid, np.nan, dtype=float)
    
    for i, (E, ra) in enumerate(zip(E_grid, r_a)):
        # Only integrate from r=0 to apocenter r_a
        # Use boolean indexing with pre-computed mask for efficiency
        mask = r_grid <= ra
        r_sub = r_grid[mask]
        if len(r_sub) < 3:
            continue # leave it as nan when not enough points to compute period
        # Interpolate potential on subset
        psi_sub = np.interp(r_sub, r_grid, psi_r)
        # Radial velocity: v_r = sqrt(2*(psi - E))
        # Pre-compute the difference to avoid repeated calculation
        psi_diff = psi_sub - E
        integrand = 1.0 / np.sqrt(np.maximum(2.0 * psi_diff, 1e-300))
        # Integrate dr/v_r from 0 to r_a (period = 2 * this integral)
        # Units: integrand [1/(km/s)], dr [pc], so integral [pc/(km/s)]
        integral = np.trapz(integrand, r_sub)
        # Period in units of pc/(km/s)
        Period[i] = period_factor * integral
    return r_a, Period, Jc, r_c

def compute_I_moments(E_grid, r_grid, fE, r_a, psi_r):
    """
    Compute the I-moments (I0, I1/2, I3/2) needed for the angular momentum diffusion coefficient.
    
    These moments appear in the orbit-averaged Fokker-Planck equation and are defined as:
    - I0(E) = integral[0 to E] f(E') dE'
    - I1/2(r, E) = integral[E to psi(r)] f(E') * sqrt(2*(psi(r) - E')) / sqrt(2*(psi(r) - E)) dE'
    - I3/2(r, E) = integral[E to psi(r)] f(E') * (2*(psi(r) - E'))^(3/2) / (2*(psi(r) - E))^(3/2) dE'
    
    These are used to compute the orbit-averaged diffusion coefficient mu_bar(E).
    
    Parameters
    ----------
    E_grid : array_like
        Grid of binding energies in (km/s)^2
    r_grid : array_like
        Radial grid in pc
    fE : array_like
        Phase space distribution function f(E) in units of (pc*km/s)^(-3) / Msun
    r_a : array_like
        Apocenter radius for each energy in pc
    psi_r : array_like
        Potential at r_grid
    
    Returns
    -------
    I0 : array
        Zeroth moment I0(E) for each energy
    r_orbit_list : list of arrays
        Radial points along each orbit (r < r_a for each energy)
    I12_list : list of arrays
        I1/2(r, E) evaluated along each orbit
    I32_list : list of arrays
        I3/2(r, E) evaluated along each orbit
    """
    # Zeroth moment: I0(E) = integral from 0 to E of f(E') dE'
    # Use cumulative_trapezoid for vectorized computation (much faster than loop)
    I0 = cumulative_trapezoid(fE, E_grid, initial=0.0)
    
    # Compute I1/2 and I3/2 along each orbit (for each binding energy)
    I12_list, I32_list, r_orbit_list = [], [], []
    for i, (E, ra) in enumerate(zip(E_grid, r_a)):
        # Evaluate moments at each radius r along the orbit (r <= r_a)
        r_sub = r_grid[r_grid <= ra]
        psi_sub = np.interp(r_sub, r_grid, psi_r)
        I12_vals = np.zeros_like(r_sub)
        I32_vals = np.zeros_like(r_sub)
        
        for j, psi_loc in enumerate(psi_sub):
            # Skip if local potential is less than orbital energy (unphysical)
            if psi_loc <= E: continue
            
            # Energy range: from orbital energy E to local potential psi(r)
            mask = (E_grid >= E) & (E_grid <= psi_loc)
            E_seg = E_grid[mask]
            if len(E_seg) < 3: 
                continue # leave it as nan when not enough points to compute I moments
                # this can happen when the energy grid is too coarse
            f_seg = fE[mask]
            
            # Kernel for the integral: 2*(psi(r) - E')
            kernel = 2.0 * (psi_loc - E_seg)
            
            # Numerator integrals
            num12 = np.trapz(f_seg * kernel**0.5, E_seg)
            num32 = np.trapz(f_seg * kernel**1.5, E_seg)
            
            # Denominator normalization: sqrt(2*(psi(r) - E))
            denom = max(2.0 * (psi_loc - E), 1e-300)
            I12_vals[j] = num12 / denom**0.5
            I32_vals[j] = num32 / denom**1.5
            
        r_orbit_list.append(r_sub)
        I12_list.append(I12_vals)
        I32_list.append(I32_vals)
    return I0, r_orbit_list, I12_list, I32_list

def loss_cone_rate_SM16(profile_type='hernquist', params=None,
    n_r = 600, n_E = 100,
    rmin_pc = 1e-4, rmax_pc = 1e3
):
    """
    Compute the TDE rate using the Stone & Metzger (2016) loss cone formalism.
    
    This function implements the calculation of the tidal disruption rate for stars
    orbiting a supermassive black hole. The method accounts for:
    - Two-body relaxation driving angular momentum diffusion
    - Loss cone physics (stars on orbits with J < J_lc get tidally disrupted)
    - Orbit-averaged Fokker-Planck equation
    
    The algorithm follows Stone & Metzger (2016) which accounts for the transition
    between empty (q << 1) and full (q >> 1) loss cone regimes. rzero is the zero-point radius of the isothermal profile.
    
    Parameters
    ----------
    profile_type : str, optional
        Type of profile to use. Choices: 'hernquist', 'cuspy_isothermal'
        (default: 'hernquist')
    profile_params : dict, optional
        Dictionary of parameters for the profile.
        For 'hernquist': {'Mstar': float, 'a': float, 'Mbh': float}
        For 'cuspy_isothermal': {'sigma': float, 'Mbh': float, 'rzero': float}
    n_r : int, optional
        Number of radial grid points (default: 600)
    n_E : int, optional
        Number of energy grid points (default: 100)
    rmin_pc : float, optional
        Minimum radius for integration in pc (default: 1e-4)
    rmax_pc : float, optional
        Maximum radius for integration in pc (default: 1e3)
    
    Returns
    -------
    rate : float
        TDE rate in events per year
    
    References
    -----------
    Stone & Metzger (2016), MNRAS, 456, 602
    """
    # Convert effective radius to Hernquist scale radius
    # Re = 1.8153 * a for Hernquist profile
    if profile_type == 'hernquist':
        params['a'] = params['Re'] / 1.8153
    elif profile_type == 'cuspy_isothermal':
        params['rh'] = G_pc_kms2_Msun * params['Mbh'] / params['sigma']**2
    
    # Set up radial grid (logarithmic spacing)
    r_grid = np.logspace(np.log10(rmin_pc), np.log10(rmax_pc), n_r)
    psi_r = profile.total_potential(r_grid, profile_type=profile_type, profile_params=params)

    # Stellar parameters (solar-type stars)
    mstar = 1 # 0.38  # Msun
    rstar = 1 * 2.25461e-8  # pc (Rsun) #0.44
    
    # Schwarzschild radius: rg = GM/c^2
    rg = G_pc_kms2_Msun * params['Mbh'] / c_kms**2  # pc
    
    # Tidal disruption radius: r_t = r_star * (Mbh/Mstar)^(1/3)
    # Stars at r < r_t get tidally disrupted
    r_t = (0.844)**(2/3) * rstar * (params['Mbh'] / mstar) ** (1.0 / 3.0)
    
    # If tidal radius is inside event horizon (2*rg), no TDEs possible
    if r_t < 2 * rg: 
        return 0.0

    # Set up energy grid from minimum to maximum potential
    # For isothermal: potential can be negative at large r, so we need to handle this carefully
    if profile_type == 'cuspy_isothermal':
        # For isothermal: Psi(r) = -2*sigma^2*ln(r/rh)
        # Psi is positive at r < rh, zero at r=rh, negative at r > rh
        # So we need to find the actual min/max after sorting
        psi_sorted = np.sort(psi_r)
        psi_min = psi_sorted[0]   # Most negative (at largest r)
        psi_max = psi_sorted[-1]  # Most positive (at smallest r)
        # For energy grid, we want energies that are accessible
        # Use linear spacing if we have negative values, otherwise log spacing
        if psi_min < 0:
            # For isothermal with negative potentials at large r:
            # Only use positive energies (bound orbits)
            # Focus on energies where most stars are (near BH where TDEs occur)
            # Use a more conservative minimum to avoid numerical issues
            E_min = max(0.001 * psi_max, 0.01)  # Avoid very small energies
            E_max = psi_max * 0.999
            # Ensure we have enough points in the energy grid
            E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_E)
        else:
            E_grid = np.logspace(np.log10(psi_min*1.0001), np.log10(psi_max*0.9999), n_E)
    else:
        # For Hernquist: potential always decreases with radius
        psi_min, psi_max = psi_r[-1], psi_r[0]  # psi decreases with radius
        E_grid = np.logspace(np.log10(psi_min*1.0001), np.log10(psi_max*0.9999), n_E)

    # Compute phase space distribution function f(E) using Eddington inversion
    # This gives the number density in phase space as a function of energy
    # For isothermal, use more smoothing due to density singularity
    if profile_type == 'cuspy_isothermal':
        # Use larger smoothing parameter for isothermal due to singularity
        # This helps stabilize the spline fit in the presence of steep density gradients
        fE = edd.eddington_inversion_from_rho(r_grid, E_grid, profile_type=profile_type, profile_params=params,
                                     ln_spline_smoothing=1e-8, force_zero_boundary=True)/mstar
    else:
        fE = edd.eddington_inversion_from_rho(r_grid, E_grid, profile_type=profile_type, profile_params=params,
                                     ln_spline_smoothing=1e-8, force_zero_boundary=True)/mstar

    # Compute orbital properties for each energy
    # Compute vc2_arr once and pass it to avoid recomputation
    vc2_arr = profile.vcirc2(r_grid, profile_type=profile_type, profile_params=params)
    r_a, Period, Jc, r_c = compute_Jc_P_E(E_grid, r_grid, psi_r, vc2_arr)
    
    # Compute I-moments needed for diffusion coefficient
    # Pass pre-computed psi_r to avoid recomputation
    I0, r_orbits, I12_list, I32_list = compute_I_moments(E_grid, r_grid, fE, r_a, psi_r)
    
    # Coulomb logarithm for two-body relaxation
    lnLambda = np.log(0.4 * params['Mbh'] / mstar)

    # Compute the orbit-averaged angular momentum diffusion coefficient mu_bar(E)
    # This describes how fast stars diffuse in angular momentum due to two-body encounters
    # Units: mu_bar has units of (angular momentum)^2 / time = (pc*km/s)^2 / (pc/(km/s)) = pc * (km/s)^3
    # Pre-compute constant factors for efficiency
    # diffusion_prefactor units: G^2 [pc^2 (km/s)^4 / Msun^2] * m^2 [Msun^2] = pc^2 (km/s)^4
    diffusion_prefactor = 32 * np.pi**2 * G_pc_kms2_Msun**2 * mstar**2 * lnLambda / 3.0
    mu_bar = np.zeros_like(E_grid)
    
    for i, (E, ra, P) in enumerate(zip(E_grid, r_a, Period)):
        if not np.isfinite(P) or P <= 0: 
            continue
            # P is nan when there is not enough points within the apocenter radius 
            # usually the TDE rate contribution is negligible       
        
        # Use boolean indexing for efficiency
        mask = r_grid <= ra
        r_sub = r_grid[mask]
        if len(r_sub) < 3: 
            continue
            # similar to the case above, when apocenter radius is small enough, the TDE rate contribution is negligible
        
        psi_sub = np.interp(r_sub, r_grid, psi_r)
        # Radial velocity at each point along orbit
        psi_diff = psi_sub - E
        vr = np.sqrt(np.maximum(2.0 * psi_diff, 1e-300))    # km/s
        
        # Diffusion coefficient kernel (Stone & Metzger 2016)
        # The factor (3*I1/2 - I3/2 + 2*I0) comes from the orbit-averaged Fokker-Planck equation
        I_factor = 3.0 * I12_list[i] - I32_list[i] + 2.0 * I0[i]
        kernel = diffusion_prefactor * r_sub**2 / Jc[i]**2 * I_factor
        
        # Orbit average: (1/P) * integral[dr/vr * kernel]
        integral = np.trapz(kernel / np.maximum(vr, 1e-300), r_sub)  
        mu_bar[i] = (2.0 / P) * integral

    # Loss cone angular momentum squared: J_lc^2 = 2*G*Mbh*r_t
    J_lc2 = 2.0 * G_pc_kms2_Msun * params['Mbh'] * r_t
    
    # Loss cone filling factor: R_lc = J_lc^2 / Jc^2
    # This measures how small the loss cone is compared to typical angular momentum
    # For isothermal profiles at high energies, Jc can become very small (proportional to r_c),
    # causing R_lc to exceed 1, which is unphysical for the Stone & Metzger formalism
    Jc2_safe = Jc**2 + 1e-300
    R_lc = J_lc2 / Jc2_safe
    
    # Cap R_lc at a maximum value (e.g., 0.99) to ensure physical validity
    # When R_lc > 1, the loss cone is larger than circular angular momentum,
    # which means the Stone & Metzger formalism is not applicable
    # These high-energy orbits are likely already inside the loss cone
    R_lc_max = 0.999  # Maximum allowed value for physical validity
    R_lc_capped = np.minimum(R_lc, R_lc_max)
    
    # Diffusion rate parameter: q(E) = mu_bar * P / R_lc
    # q << 1: empty loss cone (stars take many orbits to reach loss cone)
    # q >> 1: full loss cone (stars are lost immediately)
    R_lc_safe = np.maximum(R_lc_capped, 1e-300)
    qE = mu_bar * Period / R_lc_safe
    print('E', E_grid)
    print('q', qE)
    print('Rlc', R_lc_capped)
    
    # Effective loss cone size R0 (accounts for both regimes)
    qE_safe = np.maximum(qE, 0.0)
    R0 = np.where(qE > 1.0,
                      R_lc_capped * np.exp(-qE),
                      R_lc_capped * np.exp(-0.186 * qE - 0.824 * np.sqrt(qE_safe)))
    print('R0', R0)

    # Cap R0 to ensure it's always < 1 (loss cone filling factor must be < 1)
    # If R0 >= 1, the orbit is already fully in the loss cone
    R0 = np.minimum(R0, 0.999)
    
    # TDE flux per unit energy: F(E) = 4*pi^2 * Jc^2 * mu_bar * f(E) / ln(R0)
    # This gives the rate at which stars with energy E are consumed by the loss cone
    # Note: log(1/R0) must be positive, so R0 must be < 1
    R0_safe = np.maximum(R0, 1e-300)
    log_R0_inv = np.maximum(np.log(1.0 / R0_safe), 1e-300)
    
    print('Jc_safe', Jc2_safe)
    F_E = 4.0 * np.pi**2 * Jc2_safe * mu_bar * fE / log_R0_inv

    # Handle any numerical infinities or NaNs
    # Also filter out energies where R_lc > R_lc_max (unphysical regime)
    mask_physical = (np.isfinite(F_E)) & (~np.isnan(F_E)) & (R_lc <= R_lc_max) & (R0 < 1) & (R0 > 0)
    F_E = np.where(mask_physical, F_E, 0.0)
    
    print(F_E)
    # Total TDE rate: integral of F(E) over all energies
    rate = np.trapz(F_E, E_grid)

    return rate

"""
Eddington inversion method to compute the phase space distribution function f(E)
from a known density profile ρ(r) and potential Ψ(r).

The Eddington formula (Binney & Tremaine 2008, eq. 4.46) is:
    f(E) = (1/(sqrt(8)π²)) * [I(E) + (dρ/dΨ)|_Ψ_min / sqrt(E)]

where:
    I(E) = 2*sqrt(E) * ∫[0 to π/2] (d²ρ/dΨ²)(E*sin²θ) * sin(θ) dθ

This inversion assumes spherical symmetry and an isotropic velocity distribution.
The method works for any monotonic potential-density pair.

Unit convention:
- Length: pc
- Velocity: km/s  
- Mass: Msun
- Potential (Psi): (km/s)²
- Energy (E): (km/s)²
- Density (rho): Msun / pc³
- Distribution function f(E): number per unit volume per unit velocity³ per unit energy
  = 1 / [Msun * pc³ * (km/s)³ * (km/s)²] = 1 / [Msun * pc³ * (km/s)⁵]
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid, quad
from header import *
import density_profile as profile

def eddington_inversion_from_rho(r, E_grid, profile_type='hernquist', profile_params=None,
                                 ln_spline_smoothing=0.0, force_zero_boundary=True):
    """
    Compute the phase space distribution function f(E) using Eddington inversion.
    
    This function implements the Eddington inversion formula to recover f(E) from
    a known density profile. The method assumes spherical symmetry and isotropy.
    
    Parameters
    ----------
    r : array_like
        Radial grid in pc (monotonically increasing)
    E_grid : array_like
        Grid of binding energies E = -1/2*v² + Ψ in (km/s)²
        Should span from minimum to maximum potential values
    ln_spline_smoothing : float, optional
        Smoothing parameter for spline fit to ln(ρ) vs Ψ.
        Smaller values give less smoothing (default: 0.0 = no smoothing)
        For numerical stability, use small positive value (e.g., 1e-8)
    force_zero_boundary : bool, optional
        If True, sets boundary term (dρ/dΨ)|_Ψ_min = 0.
        This is appropriate if ρ(Ψ_min) = 0 or if dρ/dΨ → 0 at minimum potential.
        (default: True)
    profile_type : str, optional
        Type of profile to use. Choices: 'hernquist', 'cuspy_isothermal'
        (default: 'hernquist')
    profile_params : dict, optional
        Dictionary of parameters for the profile.
        For 'hernquist': {'Mstar': float, 'a': float, 'Mbh': float}
        For 'cuspy_isothermal': {'sigma': float, 'Mbh': float}
    
    Returns
    -------
    fEarray : array
        Phase space distribution function f(E) for each energy in E_grid.
        Units: 1 / [Msun * pc³ * (km/s)⁵]
        This represents the number density in phase space per unit energy.
    
    Notes
    -----
    The Eddington inversion formula requires:
    1. d²ρ/dΨ² exists (density is smooth in potential space)
    2. Potential Ψ is monotonic (decreases with radius)
    3. The integral ∫(d²ρ/dΨ²)dΨ converges
    
    The method uses a spline fit to ln(ρ) vs Ψ to ensure smooth derivatives.
    The boundary term accounts for the behavior at the minimum potential.
    
    References
    ----------
    Binney & Tremaine (2008), Galactic Dynamics, 2nd ed., Section 4.3.1
    """
    # Compute potential and density at each radius
    # Psi: potential in (km/s)²
    # rho: density in Msun / pc³
    Psi  = profile.total_potential(r, profile_type=profile_type, profile_params=profile_params)
    if profile_type == 'hernquist':
        rho  = profile.hernquist_density(r, profile_params['Mstar'], profile_params['a'])
    elif profile_type == 'cuspy_isothermal':
        # For isothermal, filter out very small radii where density singularity dominates
        # This avoids numerical instability in the Eddington inversion
        sigma = profile_params['sigma']
        # Set a maximum density threshold to avoid singularity issues
        # rho = sigma^2 / (2*pi*G*r^2), so r_min = sqrt(sigma^2 / (2*pi*G*rho_max))
        rho_max = 1e9  # Maximum density in Msun/pc^3 to avoid numerical issues
        r_min_effective = np.sqrt(sigma**2 / (2 * np.pi * G_pc_kms2_Msun * rho_max))
        # Filter out radii smaller than this threshold
        # Only filter based on the physically motivated density threshold
        mask = r >= r_min_effective
        r_filtered = r[mask]
        Psi_filtered = Psi[mask]
        rho_filtered = profile.cuspy_isothermal_density(r_filtered, sigma)
        # Use filtered arrays for spline fitting
        Psi = Psi_filtered
        rho = rho_filtered
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    # Sort by potential (ascending: Psi[0] = minimum potential at r_max)
    # This is required because the Eddington formula integrates from Ψ_min to E
    sort_idx = np.argsort(Psi)
    Psi = Psi[sort_idx]
    rho = rho[sort_idx]
    
    # Fit spline to ln(ρ) vs Ψ (working in log space for numerical stability)
    # This gives us a smooth function y(Ψ) = ln(ρ(Ψ))
    # For isothermal: rho can be very large at small r, handle numerical issues
    rho_safe = np.maximum(rho, 1e-100)  # Avoid log of zero or very small values
    y = np.log(rho_safe)
    
    # Cubic spline fit: y(Ψ) = ln(ρ(Ψ))
    # s=0 means no smoothing (interpolating spline)
    # s>0 adds smoothing which may help with noisy data
    spl = UnivariateSpline(Psi, y, s=ln_spline_smoothing, k=3)
    
    # Compute boundary term: (dρ/dΨ)|_Psi=0
    # The boundary term in the Eddington formula is: (dρ/dΨ)|_0 / sqrt(E)
    # This should be evaluated at Psi = 0, NOT at the minimum potential in the array
    # For cuspy isothermal, Psi can be negative, so we need to evaluate at Psi = 0
    if force_zero_boundary:
        # Force boundary term to zero (appropriate if ρ vanishes at Psi = 0)
        d_rho_d_Psi_at0 = 0.0
    else:
        # Evaluate boundary term at Psi = 0
        Psi0 = 0.0  # Boundary term should be evaluated at Psi = 0
        # Check if Psi = 0 is within the spline range
        if Psi[0] <= 0.0 <= Psi[-1]:
            # Psi = 0 is within the range, evaluate the spline
            yprime0 = spl.derivative(n=1)(Psi0)  # d(ln(ρ))/dΨ = (1/ρ) * dρ/dΨ
            rho0 = np.exp(spl(Psi0))  # ρ(Psi = 0)
            # dρ/dΨ = ρ * d(ln(ρ))/dΨ
            d_rho_d_Psi_at0 = rho0 * yprime0
        else:
            # Psi = 0 is outside the range, extrapolate or set to zero
            # For isothermal, if all Psi < 0, the boundary term at Psi = 0
            # should be evaluated by extrapolating the density profile
            # For now, we'll extrapolate using the spline
            try:
                yprime0 = spl.derivative(n=1)(Psi0)  # d(ln(ρ))/dΨ = (1/ρ) * dρ/dΨ
                rho0 = np.exp(spl(Psi0))  # ρ(Psi = 0)
                d_rho_d_Psi_at0 = rho0 * yprime0
            except:
                # If extrapolation fails, set boundary term to zero
                d_rho_d_Psi_at0 = 0.0

    # Pre-compute first and second derivatives of the spline for efficiency
    spl1 = spl.derivative(n=1)  # d(ln(ρ))/dΨ
    spl2 = spl.derivative(n=2)  # d²(ln(ρ))/dΨ²

    def d2rho_dPsi2(Psi_val):
        """
        Compute d²ρ/dΨ² at a given potential value.
        
        Using the chain rule:
        d²ρ/dΨ² = ρ * [(d(ln(ρ))/dΨ)² + d²(ln(ρ))/dΨ²]
        
        This is needed for the Eddington integral I(E).
        
        Parameters
        ----------
        Psi_val : float
            Potential value in (km/s)²
            
        Returns
        -------
        d2rho_dPsi2 : float
            Second derivative of density with respect to potential
            Units: (Msun / pc³) / (km/s)⁴
        """
        # For the Eddington formula, we need to evaluate at Psi = 0 even if Psi[0] < 0
        # Allow extrapolation to Psi = 0, but be more careful for values far outside the range
        # If Psi_val is far below Psi[0] (more than a small tolerance), return 0
        # But always allow evaluation at Psi = 0 (required by Eddington formula)
        if Psi_val > Psi[-1]:
            # Beyond maximum potential, return 0
            return 0.0
        if Psi_val < Psi[0] - 1e-6:
            # Far below minimum potential, return 0
            # But allow extrapolation for Psi_val between Psi[0] - 1e-6 and Psi[0]
            # and especially for Psi_val = 0 (required by Eddington formula)
            return 0.0
        
        try:
            y = spl(Psi_val)  # ln(ρ)
            yp = spl1(Psi_val)  # d(ln(ρ))/dΨ
            ypp = spl2(Psi_val)  # d²(ln(ρ))/dΨ²
            rho_val = np.exp(y)  # ρ
            # d²ρ/dΨ² = ρ * [(d(ln(ρ))/dΨ)² + d²(ln(ρ))/dΨ²]
            result = rho_val * (yp*yp + ypp)
            # Check for NaN or inf
            if not np.isfinite(result):
                return 0.0
            return result
        except:
            # If spline evaluation fails, return zero
            return 0.0

    # Determine integration parameters based on profile type
    # For isothermal, use more conservative parameters due to singularity
    if profile_type == 'cuspy_isothermal':
        quad_epsabs = 1e-6
        quad_epsrel = 1e-5
        quad_limit = 500
    else:
        quad_epsabs = 1e-8
        quad_epsrel = 1e-7
        quad_limit = 200
    
    def I_of_E(E):
        """
        Compute the integral I(E) in the Eddington formula.
        
        I(E) = 2*sqrt(E) * ∫[0 to π/2] (d²ρ/dΨ²)(E*sin²θ) * sin(θ) dθ
        
        This integral uses the substitution Ψ = E*sin²θ to transform from
        Ψ-space to θ-space, which makes the integration more stable.
        
        Parameters
        ----------
        E : float
            Binding energy in (km/s)²
            
        Returns
        -------
        I : float
            Integral value with units: (Msun / pc³) / (km/s)²
        """
        if E <= 0:
            # Unphysical (E must be positive for bound orbits)
            return 0.0
        
        # The Eddington formula integrates from Ψ = 0 to E
        # The substitution Ψ = E*sin²θ correctly integrates from 0 to E
        # For isothermal with negative potentials, Psi[0] < 0, but we still
        # integrate from Ψ = 0 (not from Psi[0])
        # Check if E is within reasonable range (E should be positive and <= Psi_max)
        if E <= 0 or E > Psi[-1]:
            return 0.0
        
        def integrand(theta):
            """
            Integrand for the Eddington integral.
            
            Using substitution Ψ = E*sin²θ:
            - When θ=0: Ψ=0 (this is correct, even if Psi[0] < 0)
            - When θ=π/2: Ψ=E (energy value we're computing for)
            
            The Eddington formula integrates from Ψ = 0 to E, not from Psi[0] to E.
            For isothermal with negative potentials, we need to allow extrapolation
            to Ψ = 0 even if Psi[0] < 0.
            """
            # Transform to potential space: Ψ = E*sin²θ
            # This correctly integrates from Ψ = 0 to Ψ = E
            Psi_val = E * (np.sin(theta)**2)
            # The Eddington formula requires integration from Ψ = 0 to E
            # Even if Psi[0] < 0, we need to evaluate at Ψ = 0
            # The d2rho_dPsi2 function will handle extrapolation if needed
            # Evaluate d²ρ/dΨ² at this potential value
            # Multiply by sin(θ) from the Jacobian of the substitution
            val = d2rho_dPsi2(Psi_val) * np.sin(theta)
            return val if np.isfinite(val) else 0.0
        
        try:
            # Integrate from θ=0 to θ=π/2
            # This corresponds to Ψ from 0 to E
            # For isothermal, if E is negative, we need different handling
            if E < 0:
                # For negative energies, the standard substitution doesn't work
                # We need to integrate from Psi_min to E
                # Use direct integration in Psi space instead
                # The Eddington formula for negative E needs special handling
                # For now, return 0 for negative energies (unbound or very weakly bound)
                return 0.0
            else:
                # Use adaptive integration parameters (set above based on profile type)
                val, err = quad(integrand, 0.0, 0.5*np.pi, 
                                epsabs=quad_epsabs, epsrel=quad_epsrel, limit=quad_limit)
                # Check if integration converged well
                if abs(err) > 0.1 * abs(val) and abs(val) > 1e-10:
                    # Integration didn't converge well, return 0 to avoid spurious results
                    print("Warning: Integration in eddington_inversion.py didn't converge well, return 0")
                    return 0.0
                # Multiply by 2*sqrt(E) as per Eddington formula
                result = 2.0 * np.sqrt(E) * val
                return result if np.isfinite(result) else 0.0
        except:
            return 0.0

    # Compute f(E) for each energy in the grid
    fEarray = np.zeros_like(E_grid, dtype=np.float64)
    # Prefactor from Eddington formula: 1/(sqrt(8)π²)
    prefac = 1.0 / (np.sqrt(8.0) * np.pi**2)
    
    for i, E in enumerate(E_grid):
        # Compute the integral I(E)
        I = I_of_E(E)
        
        # Compute boundary term: (dρ/dΨ)|_0 / sqrt(E)
        # This accounts for behavior at minimum potential
        boundary = 0.0 if force_zero_boundary else d_rho_d_Psi_at0 / np.sqrt(E)
        
        # Apply Eddington formula: f(E) = (1/(sqrt(8)π²)) * [I(E) + boundary]
        fE = prefac * (I + boundary)
        
        # Handle numerical errors: if f(E) is slightly negative due to numerical
        # precision but should be zero, set it to zero
        if fE < 0 and fE > -1e-12 * np.abs(I):
            fE = 0.0
        
        fEarray[i] = fE

    return fEarray

def spherical_potential_from_rho(r, rho, G=G_pc_kms2_Msun):
    """
    Compute gravitational potential from density profile for spherical systems.
    
    This function computes the potential using:
        Φ(r) = -G * ∫[r to ∞] (M_enc(s) / s²) ds - G * M_tot / r_max
    
    where M_enc(r) = 4π ∫[0 to r] ρ(s) s² ds is the enclosed mass.
    
    Parameters
    ----------
    r : array_like
        Radial grid in pc (monotonically increasing)
    rho : array_like
        Density profile in Msun / pc³
    G : float, optional
        Gravitational constant in units of pc (km/s)² / Msun
        (default: G_pc_kms2_Msun from header)
    
    Returns
    -------
    phi : array
        Gravitational potential in (km/s)²
        Note: This returns -Φ (the actual potential is negative)
    Menc : array
        Enclosed mass profile in Msun
        Menc[i] = mass within radius r[i]
    
    Notes
    -----
    This function is useful for density profiles that don't have analytic
    potential forms. The integration is done from r=0 to r_max.
    """
    # Compute enclosed mass: M_enc(r) = 4π ∫[0 to r] ρ(s) s² ds
    # Units: ρ [Msun/pc³] * r² [pc²] * dr [pc] = Msun
    Menc = 4*np.pi * cumulative_trapezoid(rho * r**2, r, initial=0.0)
    Mtot = Menc[-1]  # Total mass
    
    # Compute potential: Φ(r) = -G * ∫[r to ∞] (M_enc(s) / s²) ds - G * M_tot / r_max
    # To integrate from r to ∞, we reverse the arrays and integrate from ∞ to r
    s = r[::-1]  # Reversed radius array (from r_max to r_min)
    # Integrand: G * M_enc(s) / s²
    # Units: G [pc(km/s)²/Msun] * M_enc [Msun] / s² [pc²] = (km/s)² / pc
    integrand = G * (Menc[::-1] / s**2)
    # Integrate from ∞ (s[0]) to r_min (s[-1])
    I = cumulative_trapezoid(integrand, s, initial=0.0)
    # Reverse back to get integral from r to ∞
    integral_r_to_rmax = -I[::-1]
    # Add boundary term: -G * M_tot / r_max
    # Units: G [pc(km/s)²/Msun] * M_tot [Msun] / r_max [pc] = (km/s)²
    phi = -integral_r_to_rmax - G * Mtot / r[-1]
    return phi, Menc
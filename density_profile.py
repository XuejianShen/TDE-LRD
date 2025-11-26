import numpy as np
from header import *

# unit convention: pc, km/s, Msun, yr

# profile setup
def hernquist_density(r, Mstar, a):
    return Mstar / (2.0 * np.pi) * a / (r * (r + a) ** 3)

def hernquist_menclosed(r, Mstar, a):
    return Mstar * r**2 / (r + a)**2

def hernquist_fphase(eps):
    # distribution function of the Hernquist model
    # eps   : in unit of GM/a, and is the absolute value of the energy
    # taken from Binney & Tremaine 2008, Eq 4.51
    prefactor = 1./ ( np.sqrt(2)* 8*np.pi**3 )
    # return in unit of (G*M*a)**(-3/2)
    return prefactor * np.sqrt(eps)/(1-eps)**2 * ( (1-2*eps)*(8*eps**2-8*eps-3) + 
                3*np.arcsin(np.sqrt(eps))/np.sqrt(eps*(1-eps)) )


def hernquist_potential_star(r, Mstar, a):
    # negative of the actual potential
    return G_pc_kms2_Msun * Mstar / (r + a)

def bh_potential(r, Mbh):
    return G_pc_kms2_Msun * Mbh / r

# Plummer sphere
def plummer_density(r, M=1.0, a=1.0):
    return (3 * M)/(4*np.pi*a**3) * (1 + (r/a)**2)**(-2.5)

def plummer_analytic_f_of_E(E):
    coeff = 24*np.sqrt(2)/(7*np.pi**3)
    return coeff * np.maximum(E, 0.0)**(3.5)

# cuspy isothermal density profile
def cuspy_isothermal_density(r, sigma):
    """
    Cuspy isothermal density profile:
    rho(r) = (sigma^2) / (2 * pi * G * r^2)
    """
    return sigma**2 / (2 * np.pi * G_pc_kms2_Msun * np.maximum(r, 1e-300)**2)

def cuspy_isothermal_potential(r, sigma, rzero):
    """
    Potential for cuspy isothermal, with zero-point radius rzero:
    Phi(r) = -2 * sigma^2 * ln(r / rzero)
    """
    r_safe = np.maximum(r, 1e-30)
    return - 2.0 * sigma**2 * np.log(r_safe / rzero)

## total potential
def total_potential(r, profile_type='hernquist', profile_params=None):
    if profile_type == 'hernquist':
        return hernquist_potential_star(r, profile_params['Mstar'], profile_params['a']) + bh_potential(r, profile_params['Mbh'])
    elif profile_type == 'cuspy_isothermal':
        return cuspy_isothermal_potential(r, profile_params['sigma'], profile_params['rzero']) + bh_potential(r, profile_params['Mbh'])
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

def vcirc2(r, profile_type='hernquist', profile_params=None):
    # circular velocity squared 
    # used for defining Jc(eps)
    psi = total_potential(r, profile_type=profile_type, profile_params=profile_params)
    dpsi_dr = np.gradient(psi, r)
    return -r * dpsi_dr
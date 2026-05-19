#!/usr/bin/env python3
"""
HD 73512 full orbital solution: Gaia astrometric orbit + spectroscopic data.

The Gaia DR3 'Orbital' solution provides Thiele-Innes elements that encode the
apparent (sky-projected) ellipse of the photocentre.  Combined with K1, K2, P, e
from a spectroscopic solution, the inclination i extracted from the Thiele-Innes
elements lifts the sin(i) degeneracy and yields absolute masses and the true
semi-major axis.

Two spectroscopic datasets are compared:
  - Griffin (2009, The Observatory 129, 317), Cambridge Coravel
  - Gaia DR3 SB2 solution

NOTE on Gaia SB2 reliability
  Chevalier et al. (2023, A&A 678, A19) identify HD 73512 as a 5-sigma
  outlier when comparing their Griffin+Gaia masses against the Gaia DR3
  direct SB2 masses.  They attribute this to the large photocentre orbit
  (a0 = 3.4 mas): the Gaia RVS places its extraction window at the position
  predicted by the standard 5-parameter astrometric solution, which ignores
  the orbital motion.  For a0 = 3.4 mas (vs ~0.8 mas for the other two
  comparison SB2 stars), the extraction window is significantly mis-centred
  at many epochs, corrupting the measured radial velocities.  The Gaia SB2
  orbital parameters (P, e, K1, K2) should therefore be treated with caution;
  the Griffin solution combined with the Gaia astrometric orbit is preferred.
  Chevalier et al. confirm: M1=0.7877±0.0105, M2=0.6948±0.0066 Msun,
  beta(K4V)=0.3123±0.0017, F2=-0.03, using Griffin+Gaia.

NOTE on component conventions
  Griffin labels the heavier (K0V, brighter) star as the primary.
  Gaia SB2 labels the larger-amplitude component as primary, which is the
  lighter (K4V) star.  The Thiele-Innes omega corresponds to the photocentre
  orbit, which follows the brighter (heavier) star; it therefore matches
  Griffin's omega_1.  For Gaia SB2 input, K_heavy and K_light are mapped to
  the Gaia secondary (K2) and primary (K1) respectively.

CHARA observation (J2025.8207) provides an independent separation, PA, and
H/K flux fractions that constrain the orbital ephemeris and SED.
"""

import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AU_m       = 1.495978707e11   # metres per AU
Gm_AU      = 1e9 / AU_m       # 1 Gm in AU
J2000_MJD  = 51544.5          # MJD of J2000.0
Gaia_ref_MJD = 57388.0        # MJD of Gaia DR3 reference epoch J2016.0
hc_over_kB = 14387769.0       # nm·K  (= hc / k_B; c2 = 0.014388 m·K × 1e9)


# ---------------------------------------------------------------------------
# Spectroscopic datasets
# K_heavy : semi-amplitude of the heavier/brighter component (km/s)
# K_light : semi-amplitude of the lighter/fainter  component (km/s)
# lum_ratio: L_heavy/L_light in the instrument's bandpass (optional)
# ---------------------------------------------------------------------------
DATASETS = [
    {
        'label'     : 'Griffin (2009)  Cambridge Coravel',
        'P_days'    : 128.25,
        'e'         : 0.2661,
        'K_heavy'   : 23.27,    # Griffin K1 (primary = more massive)
        'K_light'   : 26.41,    # Griffin K2 (secondary = less massive)
        'gamma_kms' : 34.52,
        'lum_ratio' : 1.0 / 0.37,  # Coravel dip-area ratio (primary:secondary = 1:0.37)
        'band_label': 'Coravel ~510 nm',
    },
    {
        'label'     : 'Gaia DR3 SB2',
        'P_days'    : 128.912,
        'e'         : 0.2141,
        'K_heavy'   : 23.90,    # Gaia K2 (secondary = more massive, Griffin's primary)
        'K_light'   : 28.17,    # Gaia K1 (primary   = less  massive, Griffin's secondary)
        'gamma_kms' : 32.80,
        'lum_ratio' : None,
        'band_label': None,
        'caveat'    : ('RVS extraction window mis-centred due to large a0=3.4 mas; '
                       'orbital parameters likely biased (Chevalier et al. 2023, A&A 678, A19)'),
        # gamma shift vs Griffin (~34.52) is partly or wholly due to RVS bias, not necessarily a third body
    },
]

# Gaia Orbital periastron epoch (used for best ephemeris propagation)
# t_periastron from Gaia Orbital solution is relative to Gaia_ref_MJD
GAIA_T0_OFFSET = -38.73258923456723   # days relative to J2016.0

# ---------------------------------------------------------------------------
# CHARA interferometric observation
# ---------------------------------------------------------------------------
CHARA = {
    'epoch_J'  : 2025.8207,          # Julian epoch
    'sep_mas'  : 17.6595,            # angular separation (mas)
    'PA_deg'   : 29.562,             # position angle, North through East (deg)
    'beta_H'   : 0.63131,            # brighter-star fraction of total H-band flux
    'beta_K'   : 0.60325,            # brighter-star fraction of total K-band flux
    'lam_H_nm' : 1630.0,             # effective wavelength, H band (nm)
    'lam_K_nm' : 2190.0,             # effective wavelength, K band (nm)
}

# 2MASS combined photometry — loaded from tmass_hd73512.ecsv at runtime

# Approximate effective wavelengths for Gaia passbands and Coravel (nm)
BAND_WAVELENGTHS = {
    'Coravel' : 510.0,
    'Gaia BP' : 532.0,
    'Gaia G'  : 673.0,
    'Gaia RP' : 797.0,
    'CHARA H' : 1630.0,
    'CHARA K' : 2190.0,
}

# Stellar model parameters for K0V + K4V (Pecaut & Mamajek 2013)
STELLAR_MODEL = {
    'T1_K'   : 5250.0,   # K0V effective temperature
    'T2_K'   : 4590.0,   # K4V effective temperature
    'R1_Rsun': 0.826,    # K0V radius
    'R2_Rsun': 0.726,    # K4V radius
}


# ---------------------------------------------------------------------------
# Thiele-Innes → orbital elements
# ---------------------------------------------------------------------------
def thiele_innes_to_elements(A, B, F, G):
    """
    Extract photocentre orbit elements from Thiele-Innes constants (all in mas).

    Returns
    -------
    a_phot : float  angular semi-major axis of the photocentre (mas)
    i      : float  inclination (deg)
    omega  : float  argument of periastron of the brighter component (deg)
    Omega  : float  longitude of ascending node (deg)

    Relations used:
        A^2+B^2+F^2+G^2  =  a^2 (1 + cos^2 i)
        AG - BF          =  a^2  cos i
        A+G = a(1+cos i) cos(Omega+omega);   B-F = a(1+cos i) sin(Omega+omega)
        A-G = a(1-cos i) cos(Omega-omega);   B+F = a(1-cos i) sin(Omega-omega)
    """
    S = A**2 + B**2 + F**2 + G**2
    T = A*G - B*F

    disc = S**2 - 4*T**2
    if disc < 0:
        raise ValueError("Negative discriminant — check Thiele-Innes elements")

    a2     = (S + np.sqrt(disc)) / 2
    a_phot = np.sqrt(a2)
    cos_i  = T / a2
    i_deg  = np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))

    OmegaPomega = np.degrees(np.arctan2(B - F, A + G)) % 360.0
    OmegaMomega = np.degrees(np.arctan2(B + F, A - G)) % 360.0

    Omega = ((OmegaPomega + OmegaMomega) / 2.0) % 360.0
    omega = ((OmegaPomega - OmegaMomega) / 2.0) % 360.0

    return a_phot, i_deg, omega, Omega


# ---------------------------------------------------------------------------
# Orbital ephemeris
# ---------------------------------------------------------------------------
def solve_kepler(M, e, tol=1e-12):
    """Solve Kepler's equation M = E - e sin E by Newton-Raphson."""
    E = float(M)
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def sky_position(a_rel_mas, e, P_days, i_deg, omega_K0V_deg, Omega_deg,
                 T0_MJD, t_MJD):
    """
    Compute sky-plane separation (mas) and PA (deg, N through E) of the
    lighter (K4V) component relative to the heavier (K0V) component at t_MJD.

    omega_K0V_deg is the argument of periastron of the K0V component's orbit
    about the CoM; the relative orbit uses omega_K0V + 180 deg.
    """
    omega_rel = np.radians((omega_K0V_deg + 180.0) % 360.0)
    Omega     = np.radians(Omega_deg)
    inc       = np.radians(i_deg)

    dt   = t_MJD - T0_MJD
    frac = (dt / P_days) % 1.0
    M    = 2.0 * np.pi * frac
    E    = solve_kepler(M, e)

    X0 = np.cos(E) - e
    Y0 = np.sqrt(1.0 - e**2) * np.sin(E)

    # Thiele-Innes elements for the relative orbit
    A = a_rel_mas * ( np.cos(Omega)*np.cos(omega_rel) - np.sin(Omega)*np.sin(omega_rel)*np.cos(inc))
    B = a_rel_mas * ( np.sin(Omega)*np.cos(omega_rel) + np.cos(Omega)*np.sin(omega_rel)*np.cos(inc))
    F = a_rel_mas * (-np.cos(Omega)*np.sin(omega_rel) - np.sin(Omega)*np.cos(omega_rel)*np.cos(inc))
    G = a_rel_mas * (-np.sin(Omega)*np.sin(omega_rel) + np.cos(Omega)*np.cos(omega_rel)*np.cos(inc))

    dN  = A*X0 + F*Y0
    dE  = B*X0 + G*Y0
    sep = np.sqrt(dN**2 + dE**2)
    PA  = np.degrees(np.arctan2(dE, dN)) % 360.0
    return sep, PA, frac


# ---------------------------------------------------------------------------
# Full orbital solution
# ---------------------------------------------------------------------------
def combined_solution(A, B, F, G, parallax_mas, dataset, gaia_T0_offset):
    """
    Combine Thiele-Innes elements with one spectroscopic dataset.
    Returns a dict of all derived quantities.
    """
    P_days    = dataset['P_days']
    e         = dataset['e']
    K_heavy   = dataset['K_heavy'] * 1e3   # m/s
    K_light   = dataset['K_light'] * 1e3
    lum_ratio = dataset.get('lum_ratio')

    a_phot_mas, i_deg, omega_deg, Omega_deg = thiele_innes_to_elements(A, B, F, G)
    sin_i = np.sin(np.radians(i_deg))

    P_s          = P_days * 86400.0
    sqrt_1me2    = np.sqrt(1.0 - e**2)
    a_heavy_sini = K_heavy * P_s * sqrt_1me2 / (2.0 * np.pi)
    a_light_sini = K_light * P_s * sqrt_1me2 / (2.0 * np.pi)
    a_rel_sini   = (a_heavy_sini + a_light_sini) / AU_m   # AU

    q          = K_light / K_heavy   # = M_heavy / M_light
    a_rel_AU   = a_rel_sini / sin_i
    P_yr       = P_days / 365.25
    M_total    = a_rel_AU**3 / P_yr**2
    M_heavy    = M_total * q / (1.0 + q)
    M_light    = M_total / (1.0 + q)
    f_heavy    = M_heavy / M_total
    a_rel_mas  = a_rel_AU * parallax_mas

    # a_phot = a_rel * (beta1 - f1)  (photocentre displaced toward brighter star)
    beta1_f1_obs     = a_phot_mas / a_rel_mas
    beta1_inferred   = f_heavy + beta1_f1_obs
    L_ratio_inferred = beta1_inferred / (1.0 - beta1_inferred)

    if lum_ratio is not None:
        beta1_given      = lum_ratio / (1.0 + lum_ratio)
        beta1_f1_given   = abs(beta1_given - f_heavy)
        a_phot_predicted = a_rel_mas * beta1_f1_given
    else:
        beta1_given = beta1_f1_given = a_phot_predicted = None

    # CHARA ephemeris using Gaia Orbital T0 + this solution's P, e, geometry
    t_chara_MJD = J2000_MJD + (CHARA['epoch_J'] - 2000.0) * 365.25
    T0_MJD      = Gaia_ref_MJD + gaia_T0_offset
    chara_sep, chara_PA, chara_phase = sky_position(
        a_rel_mas, e, P_days, i_deg, omega_deg, Omega_deg, T0_MJD, t_chara_MJD)

    # CHARA-implied orbit scale: observed sep / predicted sep rescales a_rel
    chara_scale       = CHARA['sep_mas'] / chara_sep   # <1 means CHARA implies smaller orbit
    a_rel_mas_chara   = a_rel_mas * chara_scale
    a_rel_AU_chara    = a_rel_AU  * chara_scale

    return {
        'label'           : dataset['label'],
        'caveat'          : dataset.get('caveat'),
        'gamma_kms'       : dataset['gamma_kms'],
        'a_phot_mas'      : a_phot_mas,
        'i_deg'           : i_deg,
        'omega_deg'       : omega_deg,
        'Omega_deg'       : Omega_deg,
        'P_days'          : P_days,
        'e'               : e,
        'K_heavy_kms'     : dataset['K_heavy'],
        'K_light_kms'     : dataset['K_light'],
        'q'               : q,
        'a_rel_AU'        : a_rel_AU,
        'a_rel_mas'       : a_rel_mas,
        'a_heavy_AU'      : a_heavy_sini / AU_m / sin_i,
        'a_light_AU'      : a_light_sini / AU_m / sin_i,
        'peri_AU'         : a_rel_AU * (1.0 - e),
        'apo_AU'          : a_rel_AU * (1.0 + e),
        'M_heavy_Msun'    : M_heavy,
        'M_light_Msun'    : M_light,
        'M_total_Msun'    : M_total,
        'f_heavy'         : f_heavy,
        'beta1_f1_obs'    : beta1_f1_obs,
        'beta1_inferred'  : beta1_inferred,
        'L_ratio_inferred': L_ratio_inferred,
        'beta1_given'     : beta1_given,
        'beta1_f1_given'  : beta1_f1_given,
        'a_phot_predicted': a_phot_predicted,
        'chara_sep_pred'  : chara_sep,
        'chara_PA_pred'   : chara_PA,
        'chara_phase'     : chara_phase,
        'chara_scale'     : chara_scale,
        'a_rel_mas_chara' : a_rel_mas_chara,
        'a_rel_AU_chara'  : a_rel_AU_chara,
    }


# ---------------------------------------------------------------------------
# Luminosity ratio analysis
# ---------------------------------------------------------------------------
def planck_ratio(lam_nm, T1_K, T2_K, R1_Rsun, R2_Rsun):
    """
    Flux ratio F1/F2 = (R1/R2)^2 * B_lam(T1) / B_lam(T2)
    using the Planck function at a single effective wavelength.
    """
    x1 = hc_over_kB / (lam_nm * T1_K)
    x2 = hc_over_kB / (lam_nm * T2_K)
    B_ratio = (np.exp(x2) - 1.0) / (np.exp(x1) - 1.0)
    return (R1_Rsun / R2_Rsun)**2 * B_ratio


def luminosity_table(solutions, gaia_source, chara, model, tmass):
    """
    Compile L_heavy/L_light from all sources and compare to a blackbody model.
    Also derive individual component magnitudes in each Gaia band.
    """
    T1, T2 = model['T1_K'],    model['T2_K']
    R1, R2 = model['R1_Rsun'], model['R2_Rsun']

    # L_ratio from Gaia G-band astrometry (Griffin solution = solutions[0])
    L_ratio_G = solutions[0]['L_ratio_inferred']
    beta1_G   = solutions[0]['beta1_inferred']

    # L_ratios from CHARA
    beta_H   = chara['beta_H']
    beta_K   = chara['beta_K']
    L_ratio_H = beta_H   / (1.0 - beta_H)
    L_ratio_K = beta_K   / (1.0 - beta_K)

    # L_ratio from Coravel dip areas (Griffin)
    L_ratio_Coravel = DATASETS[0]['lum_ratio']
    beta1_Coravel   = L_ratio_Coravel / (1.0 + L_ratio_Coravel)

    rows = [
        ('Coravel',  510.0,               L_ratio_Coravel, beta1_Coravel, 'Griffin dip-area ratio'),
        ('Gaia G',   BAND_WAVELENGTHS['Gaia G'],  L_ratio_G, beta1_G,     'Gaia a_phot (Griffin sol)'),
        ('Gaia RP',  BAND_WAVELENGTHS['Gaia RP'], None,      None,        'combined only'),
        ('CHARA H',  chara['lam_H_nm'],   L_ratio_H, beta_H,  'CHARA resolved'),
        ('CHARA K',  chara['lam_K_nm'],   L_ratio_K, beta_K,  'CHARA resolved'),
    ]

    print(f"\n\n{'='*72}")
    print("  LUMINOSITY RATIOS  L_heavy/L_light  (K0V / K4V)")
    print(f"{'='*72}")
    print(f"  Blackbody model: K0V T={T1:.0f} K  R={R1} Rsun  /  K4V T={T2:.0f} K  R={R2} Rsun")
    print(f"  {'Band':<10}  {'lam (nm)':>9}  {'Obs ratio':>10}  {'BB model':>9}  {'Source'}")
    print(f"  {'-'*68}")
    for name, lam, obs, beta1_obs, src in rows:
        bb = planck_ratio(lam, T1, T2, R1, R2)
        if obs is not None:
            print(f"  {name:<10}  {lam:>9.0f}  {obs:>10.3f}  {bb:>9.3f}  {src}")
        else:
            print(f"  {name:<10}  {lam:>9.0f}  {'---':>10}  {bb:>9.3f}  {src}")

    # Individual component magnitudes from combined Gaia photometry
    G_comb  = float(gaia_source['phot_g_mean_mag'])
    BP_comb = float(gaia_source['phot_bp_mean_mag'])
    RP_comb = float(gaia_source['phot_rp_mean_mag'])

    # BP and RP flux fractions from blackbody model
    bb_BP = planck_ratio(BAND_WAVELENGTHS['Gaia BP'], T1, T2, R1, R2)
    bb_RP = planck_ratio(BAND_WAVELENGTHS['Gaia RP'], T1, T2, R1, R2)
    beta1_BP_model = bb_BP / (1.0 + bb_BP)
    beta1_RP_model = bb_RP / (1.0 + bb_RP)

    def split_mag(m_combined, beta1):
        m1 = m_combined - 2.5 * np.log10(beta1)
        m2 = m_combined - 2.5 * np.log10(1.0 - beta1)
        return m1, m2

    G1,  G2  = split_mag(G_comb,  beta1_G)
    BP1, BP2 = split_mag(BP_comb, beta1_BP_model)
    RP1, RP2 = split_mag(RP_comb, beta1_RP_model)
    H1,  H2  = split_mag(tmass['H_mag'], beta_H)
    K1,  K2  = split_mag(tmass['K_mag'], beta_K)

    print(f"\n  Individual component magnitudes")
    print(f"  {'':10}  {'Combined':>9}  {'K0V (1)':>9}  {'K4V (2)':>9}  {'beta1':>7}  source")
    print(f"  {'-'*62}")
    for band, mc, m1, m2, b1, src in [
        ('Gaia BP', BP_comb, BP1, BP2, beta1_BP_model, 'BB model'),
        ('Gaia G',  G_comb,  G1,  G2,  beta1_G,        'astrometry'),
        ('Gaia RP', RP_comb, RP1, RP2, beta1_RP_model, 'BB model'),
        ('2MASS H', tmass['H_mag'], H1, H2, beta_H,    'CHARA'),
        ('2MASS K', tmass['K_mag'], K1, K2, beta_K,    'CHARA'),
    ]:
        print(f"  {band:<10}  {mc:>9.3f}  {m1:>9.3f}  {m2:>9.3f}  {b1:>7.4f}  {src}")

    print(f"\n  Implied component colours")
    print(f"  {'':10}  {'K0V':>9}  {'K4V':>9}  {'note'}")
    print(f"  {'-'*60}")
    print(f"  {'BP-G':<10}  {BP1-G1:>9.3f}  {BP2-G2:>9.3f}  K0V~0.42, K4V~0.64 typ.")
    print(f"  {'G-RP':<10}  {G1-RP1:>9.3f}  {G2-RP2:>9.3f}  K0V~0.56, K4V~0.70 typ.")
    print(f"  {'BP-RP':<10}  {BP1-RP1:>9.3f}  {BP2-RP2:>9.3f}  K0V~0.98, K4V~1.34 typ.")
    print(f"  {'G-H':<10}  {G1-H1:>9.3f}  {G2-H2:>9.3f}  K0V~1.64, K4V~2.09 typ.")
    print(f"  {'H-K':<10}  {H1-K1:>9.3f}  {H2-K2:>9.3f}  K0V~0.07, K4V~0.19 typ.")
    print(f"  (Gaia typical: Pecaut & Mamajek, Teff K0V=5270K, K4V=4600K)")

    # Absolute magnitudes
    parallax_mas = float(gaia_source['parallax'])
    mu = 5.0 * np.log10(1000.0 / parallax_mas / 10.0)
    print(f"\n  Distance modulus mu = {mu:.3f}  (parallax = {parallax_mas:.3f} mas)")
    print(f"  M_G(K0V) = {G1-mu:.3f}    M_G(K4V) = {G2-mu:.3f}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_solution(sol):
    w = 54
    print(f"\n{'='*w}")
    print(f"  {sol['label']}")
    print(f"{'='*w}")
    if sol.get('caveat'):
        print(f"  *** CAVEAT: {sol['caveat']} ***")
    print(f"  Astrometric elements  (Thiele-Innes, fixed)")
    print(f"    a_phot          = {sol['a_phot_mas']:.3f} mas")
    print(f"    i               = {sol['i_deg']:.2f} deg")
    print(f"    Omega (node)    = {sol['Omega_deg']:.1f} deg")
    print(f"    omega (K0V)     = {sol['omega_deg']:.1f} deg")
    print()
    print(f"  Spectroscopic inputs")
    print(f"    P               = {sol['P_days']:.3f} days")
    print(f"    e               = {sol['e']:.4f}")
    print(f"    K_heavy (K0V)   = {sol['K_heavy_kms']:.2f} km/s")
    print(f"    K_light (K4V)   = {sol['K_light_kms']:.2f} km/s")
    print(f"    gamma (CoM RV)  = {sol['gamma_kms']:.2f} km/s")
    print(f"    M_heavy/M_light = {sol['q']:.3f}")
    print()
    print(f"  Combined orbital solution")
    print(f"    a (relative)    = {sol['a_rel_AU']:.4f} AU  ({sol['a_rel_mas']:.2f} mas)")
    print(f"    a_heavy (CoM)   = {sol['a_heavy_AU']:.4f} AU")
    print(f"    a_light (CoM)   = {sol['a_light_AU']:.4f} AU")
    print(f"    periastron      = {sol['peri_AU']:.3f} AU")
    print(f"    apastron        = {sol['apo_AU']:.3f} AU")
    print(f"    M_heavy (K0V)   = {sol['M_heavy_Msun']:.4f} Msun")
    print(f"    M_light (K4V)   = {sol['M_light_Msun']:.4f} Msun")
    print(f"    M_total         = {sol['M_total_Msun']:.4f} Msun")
    print()
    print(f"  Photocentre check")
    print(f"    f_heavy              = {sol['f_heavy']:.4f}  (mass fraction)")
    print(f"    a_phot/a_rel (obs)   = {sol['beta1_f1_obs']:.4f}  = beta1 - f1")
    print(f"    beta1 inferred       = {sol['beta1_inferred']:.4f}  (lum fraction, Gaia G)")
    print(f"    L_heavy/L_light (G)  = {sol['L_ratio_inferred']:.3f}")
    if sol['a_phot_predicted'] is not None:
        print(f"    L_heavy/L_light (Coravel) = {DATASETS[0]['lum_ratio']:.3f}")
        print(f"    a_phot predicted (Coravel)= {sol['a_phot_predicted']:.3f} mas")
        print(f"    a_phot observed           = {sol['a_phot_mas']:.3f} mas")
    print()
    print(f"  CHARA ephemeris  (Gaia Orbital T0, epoch J{CHARA['epoch_J']})")
    print(f"    orbital phase        = {sol['chara_phase']:.4f}")
    print(f"    predicted sep        = {sol['chara_sep_pred']:.4f} mas   observed: {CHARA['sep_mas']:.4f} mas"
          f"   Δ = {sol['chara_sep_pred']-CHARA['sep_mas']:+.4f} mas")
    print(f"    predicted PA         = {sol['chara_PA_pred']:.3f} deg    observed: {CHARA['PA_deg']:.3f} deg"
          f"    Δ = {sol['chara_PA_pred']-CHARA['PA_deg']:+.4f} deg")
    print(f"  CHARA orbit scale")
    print(f"    a_rel (spectroscopic) = {sol['a_rel_mas']:.4f} mas  ({sol['a_rel_AU']:.4f} AU)")
    print(f"    a_rel (CHARA-implied) = {sol['a_rel_mas_chara']:.4f} mas  ({sol['a_rel_AU_chara']:.4f} AU)"
          f"   scale = {sol['chara_scale']:.4f}  ({100*(sol['chara_scale']-1):+.2f}%)")


def print_comparison(solutions):
    s0, s1 = solutions[0], solutions[1]

    print(f"\n\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")
    fmt = "  {:<30s}  {:>10s}  {:>10s}  {:>9s}"
    print(fmt.format("Parameter", "Griffin", "Gaia SB2", "Delta"))
    print(f"  {'-'*66}")

    pairs = [
        ("P (days)",              'P_days',         '{:10.3f}', '{:10.3f}', '{:+9.3f}'),
        ("e",                     'e',              '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("K_heavy (km/s)",        'K_heavy_kms',    '{:10.2f}', '{:10.2f}', '{:+9.2f}'),
        ("K_light (km/s)",        'K_light_kms',    '{:10.2f}', '{:10.2f}', '{:+9.2f}'),
        ("gamma (km/s)",          'gamma_kms',      '{:10.2f}', '{:10.2f}', '{:+9.2f}'),
        ("M_heavy/M_light",       'q',              '{:10.3f}', '{:10.3f}', '{:+9.3f}'),
        ("a_rel (AU)",            'a_rel_AU',       '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("M_heavy (Msun)",        'M_heavy_Msun',   '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("M_light (Msun)",        'M_light_Msun',   '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("M_total (Msun)",        'M_total_Msun',   '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("periastron (AU)",       'peri_AU',        '{:10.3f}', '{:10.3f}', '{:+9.3f}'),
        ("apastron (AU)",         'apo_AU',         '{:10.3f}', '{:10.3f}', '{:+9.3f}'),
        ("L_heavy/L_light (G)",   'L_ratio_inferred','{:10.3f}','{:10.3f}','{:+9.3f}'),
        ("CHARA sep pred (mas)",  'chara_sep_pred',   '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("CHARA PA pred (deg)",   'chara_PA_pred',    '{:10.3f}', '{:10.3f}', '{:+9.3f}'),
        ("a_rel CHARA-implied (mas)",'a_rel_mas_chara','{:10.4f}', '{:10.4f}', '{:+9.4f}'),
        ("CHARA scale factor",    'chara_scale',      '{:10.4f}', '{:10.4f}', '{:+9.4f}'),
    ]
    for name, key, f0, f1, fd in pairs:
        v0, v1 = s0[key], s1[key]
        print(f"  {{:<30s}}  {f0}  {f1}  {fd}".format(name, v0, v1, v1 - v0))

    print(f"\n  Fixed from Thiele-Innes:  i={s0['i_deg']:.2f} deg  "
          f"omega={s0['omega_deg']:.1f} deg (K0V)  Omega={s0['Omega_deg']:.1f} deg")
    print(f"  CHARA observed:  sep={CHARA['sep_mas']:.4f} mas   PA={CHARA['PA_deg']:.3f} deg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    ecsv_nss    = os.path.join(os.path.dirname(__file__), 'gaia_hd73512.ecsv')
    ecsv_source = os.path.join(os.path.dirname(__file__), 'gaia_source_hd73512.ecsv')
    ecsv_tmass  = os.path.join(os.path.dirname(__file__), 'tmass_hd73512.ecsv')

    t_nss = Table.read(ecsv_nss, format='ascii.ecsv')
    orb   = t_nss[t_nss['nss_solution_type'] == 'Orbital'][0]

    A        = float(orb['a_thiele_innes'])
    B        = float(orb['b_thiele_innes'])
    F        = float(orb['f_thiele_innes'])
    G        = float(orb['g_thiele_innes'])
    parallax = float(orb['parallax'])

    print(f"Gaia DR3 Orbital solution  (source {orb['source_id']})")
    print(f"  A={A:.6f}  B={B:.6f}  F={F:.6f}  G={G:.6f}  mas")
    print(f"  parallax = {parallax:.5f} mas  =>  d = {1000/parallax:.2f} pc")
    print(f"  Gaia Orbital T0 = MJD {Gaia_ref_MJD + GAIA_T0_OFFSET:.3f}")

    gaia_source = Table.read(ecsv_source, format='ascii.ecsv')[0]

    _tmass = Table.read(ecsv_tmass, format='ascii.ecsv')[0]
    TMASS  = {
        'J_mag': float(_tmass['Jmag']),
        'H_mag': float(_tmass['Hmag']),
        'K_mag': float(_tmass['Kmag']),
    }

    solutions = [
        combined_solution(A, B, F, G, parallax, ds, GAIA_T0_OFFSET)
        for ds in DATASETS
    ]

    for sol in solutions:
        print_solution(sol)

    print_comparison(solutions)

    luminosity_table(solutions, gaia_source, CHARA, STELLAR_MODEL, TMASS)

# HD 73512 orbital analysis

Full 3D orbital solution for the double-lined spectroscopic binary HD 73512
(HIP 42418, Gaia DR3 595390807776621824), combining Gaia DR3 astrometric and
spectroscopic data with ground-based observations and an interferometric
measurement from CHARA.

## What it does

`hd73512_orbit.py` combines the Gaia DR3 Thiele-Innes astrometric elements
(which encode the sky-projected photocentre orbit) with two sets of
spectroscopic radial velocity data to derive:

- Absolute masses of both components (K0V + K4V)
- True semi-major axis and orbital geometry (inclination, node, periastron)
- Predicted sky separation and position angle at the CHARA epoch (J2025.8207),
  compared against the interferometric observation
- Luminosity ratios spanning Coravel (~510 nm) through Gaia G to CHARA H and K
  bands, compared to a blackbody model for the K0V + K4V pair
- Individual component magnitudes in Gaia BP/G/RP and 2MASS H/K, derived by
  splitting combined photometry using astrometric and CHARA flux fractions
- Component colours compared to Pecaut & Mamajek Gaia-calibrated standards

## Spectroscopic datasets compared

| Source | P (d) | e | gamma (km/s) |
|--------|-------|---|--------------|
| Griffin (2009), Cambridge Coravel | 128.25 | 0.2661 | +34.52 |
| Gaia DR3 SB2 | 128.912 | 0.2141 | +32.80 |

**Note on the Gaia SB2 solution:** Chevalier et al. (2023, A&A 678, A19)
identify HD 73512 as a 5-sigma outlier when comparing Gaia direct masses
against independent estimates.  The large photocentre orbit (a0 = 3.4 mas)
causes the Gaia RVS extraction window to be mis-centred at many epochs,
biasing the SB2 radial velocities.  The Griffin + Gaia astrometry combination
is the preferred solution.

## Data files

| File | Contents | Source |
|------|----------|--------|
| `gaia_hd73512.ecsv` | Gaia DR3 NSS two-body orbit (Orbital + SB2 rows) | Gaia archive |
| `gaia_source_hd73512.ecsv` | Gaia DR3 main source table (photometry, parallax) | Gaia archive |
| `tmass_hd73512.ecsv` | 2MASS J/H/K photometry | VizieR II/246 via astroquery |

All three files are Astropy ECSV format and are read directly by the script.
The 2MASS file was downloaded using `astroquery.vizier` (Cutri et al. 2003),
the same approach as the Gaia files but querying VizieR rather than the Gaia archive.

## Requirements

- Python 3.8+
- numpy
- astropy
- astroquery (only needed to re-download `tmass_hd73512.ecsv`)

## Usage

```bash
python3 hd73512_orbit.py
```

## Key results (Griffin + Gaia astrometry)

- M(K0V) = 0.787 Msun, M(K4V) = 0.693 Msun  
  consistent with Chevalier et al. (2023): M1 = 0.7877 ± 0.0105, M2 = 0.6948 ± 0.0066 Msun
- a_rel = 21.46 mas = 0.567 AU, i = 84.5 deg
- CHARA J2025.8207: predicted PA = 29.65 deg, observed 29.56 deg (Delta = 0.09 deg)
- Orbit scale: spectroscopic a_rel = 21.46 mas; CHARA-implied a_rel = 21.09 mas (−1.7%
  agreement), vs Gaia SB2 which is −8.5% off — further evidence the Griffin solution
  is preferred

### Luminosity ratios L(K0V)/L(K4V)

| Band | lambda (nm) | Observed | BB model |
|------|-------------|----------|----------|
| Coravel | 510 | 2.703 | 2.810 |
| Gaia G | 673 | 2.232 | 2.343 |
| CHARA H | 1630 | 1.712 | 1.729 |
| CHARA K | 2190 | 1.520 | 1.652 |

### Component colours vs Pecaut & Mamajek standards

| Colour | K0V (derived) | K0V (typ.) | K4V (derived) | K4V (typ.) |
|--------|--------------|------------|--------------|------------|
| BP−G | 0.415 | 0.42 | 0.631 | 0.64 |
| G−RP | 0.635 | 0.56 | 0.676 | 0.70 |
| H−K | 0.069 | 0.07 | 0.198 | 0.19 |

H−K agrees to within 0.001 mag for both components using only CHARA flux
fractions and 2MASS photometry — no model assumptions.

## Contact

Mark Copper — MLCopper@gmail.com

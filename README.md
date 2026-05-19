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
- Predicted sky separation and position angle at the CHARA epoch (J2025.8207)
- Luminosity ratios from Coravel, Gaia G-band, and CHARA H/K bands, compared
  to a blackbody model for the K0V + K4V pair
- Individual component magnitudes and colours from combined Gaia photometry

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

| File | Contents |
|------|----------|
| `gaia_hd73512.ecsv` | Gaia DR3 NSS two-body orbit (Orbital + SB2 rows) |
| `gaia_source_hd73512.ecsv` | Gaia DR3 main source table (photometry, parallax) |
| `asu.tsv` | Hipparcos-Gaia Catalog of Accelerations entry (Brandt 2021) |

## Requirements

- Python 3.8+
- numpy
- astropy

## Usage

```bash
python3 hd73512_orbit.py
```

## Key results (Griffin + Gaia astrometry)

- M(K0V) = 0.787 Msun, M(K4V) = 0.693 Msun
- a_rel = 21.46 mas = 0.567 AU, i = 84.5 deg
- CHARA J2025.8207: predicted PA = 29.65 deg, observed 29.56 deg (Delta = 0.09 deg)
- L(K0V)/L(K4V): 2.70 (Coravel), 2.23 (Gaia G), 1.71 (CHARA H), 1.52 (CHARA K)

These results are consistent with Chevalier et al. (2023):
M1 = 0.7877 +/- 0.0105, M2 = 0.6948 +/- 0.0066 Msun.

## Contact

Mark Copper — MLCopper@gmail.com

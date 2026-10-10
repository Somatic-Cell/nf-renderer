# TSIS-1 HSRS data attribution and license

**Dataset:** Version 2 of the Total and Spectral Solar Irradiance Sensor-1
Hybrid Solar Reference Spectrum (TSIS-1 HSRS), 1 nm spectral resolution,
0.1 nm sampling interval, source file dated 2022-11-30.

**Publisher:** Laboratory for Atmospheric and Space Physics (LASP), University
of Colorado Boulder. Accessed through the LASP Interactive Solar Irradiance
Datacenter (LISIRD): https://lasp.colorado.edu/lisird/

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

The license and version are explicitly provided by the official LISIRD catalog:
https://lasp.colorado.edu/lisird/latis/catalog
The preserved `raw/TSIS_HSRS_v2_catalog_entry.json` contains the corresponding
dataset entry, including the 1 nm variant's source link and the license object.
This is the dataset license, independently of any journal article's license.

Scientific attribution:

Coddington, O. M., Richard, E. C., Harber, D., Pilewskie, P., Woods, T. N.,
Snow, M., et al. (2023). *Version 2 of the TSIS-1 Hybrid Solar Reference Spectrum
and Extension to the Full Spectrum.* Earth and Space Science, 10,
e2022EA002637. https://doi.org/10.1029/2022EA002637

Dataset description: https://lasp.colorado.edu/lisird/data/tsis1_hsrs/

Original source: https://lasp.colorado.edu/lisird/resources/lasp/hsrs/v2/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc

CSV service: https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.csv

Changes made by this package:

- The raw CSV, NetCDF, DDS, and DAS files are preserved byte for byte.
- `solar_am0.csv` selects 360..830 nm and the wavelength/irradiance columns,
  retaining their original decimal strings. It adds a header and uses LF.
- The renderer converts these values to IEEE binary32 and interpolates linearly.
  No spectral resampling, irradiance scaling, blackbody fitting, or atmosphere
  transmission has been applied during preparation.
- The spectrum is the reference at 1 AU; it is not a time-varying solar model.

Preserve this attribution, license link, and change notice when redistributing
the source or selected spectrum. No endorsement by LASP or the authors is implied.

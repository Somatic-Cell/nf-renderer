# CIE data attribution and license

The CIE color matching functions and D65 illuminant data in this package are
licensed under **Creative Commons Attribution-ShareAlike 4.0 International**:
https://creativecommons.org/licenses/by-sa/4.0/

Attribution:

- CIE 2019, *Colour-matching functions of CIE 1931 standard colorimetric observer*,
  International Commission on Illumination (CIE), Vienna, AT.
  DOI: https://doi.org/10.25039/CIE.DS.xvudnb9b
  Source: https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
- CIE 2019, *CIE standard illuminant D65*, International Commission on
  Illumination (CIE), Vienna, AT.
  DOI: https://doi.org/10.25039/CIE.DS.hjfjmt59
  Source: https://www.cie.co.at/datatable/cie-standard-illuminant-d65

The original CSV and CIE metadata files under `raw/` are preserved byte for byte.
The source metadata explicitly declares the CC BY-SA 4.0 license and contains
the published checksums. No endorsement by CIE is implied.

Changes made by this package:

- `cie_1931_2deg.csv`: added a machine-readable header and changed CRLF to LF;
  source numeric strings are unchanged.
- `d65_normalized.csv`: selected 360..830 nm and multiplied the source relative
  D65 values by the factor recorded in `manifest.json`. It makes
  `trapz(D65_normalized*ybar)/trapz(ybar) = 1` on that fixed interval. The source
  table and all normalization details are retained.
- Host/GPU representations convert the source decimal values to IEEE binary32.

These adapted CIE data files remain under CC BY-SA 4.0. Preserve this attribution,
license link, and change notice when redistributing the adapted data. The CIE
data license is separate from the renderer's source-code license and the
RGB2Spec BSD-3-Clause license.

#!/usr/bin/env python3
"""Prepare/verify real CIE, TOA solar, and official RGB2Spec assets.

Python 3.9+ standard library only. There are no analytic/synthetic substitutes.
Downloads are pinned by SHA-256; a changed upstream response is an error.
The bundled resolution-64 coefficient table can be used without a compiler.
To regenerate it explicitly, use --rebuild-model (CMake and a C++ compiler).
"""

from __future__ import annotations

import argparse
import array
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request


UPSTREAM_COMMIT = "721145dedf2491851bd46ab8fd165955cb38ddaf"
BUNDLED_MODEL_SHA256 = "1bb2e8bc757bdd90f2a8ae74e08011906697fcfa850e6e5534861577a9befeff"
SOURCES = [
    {
        "file": "CIE_xyz_1931_2deg.csv",
        "url": "https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv",
        "sha256": "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1",
        "license": "CC-BY-SA-4.0",
    },
    {
        "file": "CIE_xyz_1931_2deg.csv_metadata.json",
        "url": "https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv_metadata.json",
        "sha256": "03abcaecf4e63d77045ef57c4514b52c8bb1a46dd18e1f93a50044a0f4f481c8",
        "license": "CC-BY-SA-4.0",
    },
    {
        "file": "tsis1_hsrs_1nm.csv",
        "url": "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.csv",
        "sha256": "1cf3b07e6ac9669c429ad7ce9e92d50dfd741422efcfffa3d1e0eeb5f901616f",
        "license": "CC-BY-4.0",
    },
    {
        "file": "tsis1_hsrs_1nm.das",
        "url": "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.das",
        "sha256": "d0015b9ec62d8fcca744340e00ef3a7252b1be9b72f2870aa991d5f12d49108f",
        "license": "CC-BY-4.0",
    },
    {
        "file": "tsis1_hsrs_1nm.dds",
        "url": "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.dds",
        "sha256": "06cf52880e2c3eebb9af282c3e16bf7a33c52c91d7ad7f8e5826701a4806d332",
        "license": "CC-BY-4.0",
    },
    {
        "file": "CIE_std_illum_D65.csv",
        "url": "https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv",
        "sha256": "e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb",
        "license": "CC-BY-SA-4.0",
    },
    {
        "file": "CIE_std_illum_D65.csv_metadata_v2.json",
        "url": "https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv_metadata_v2.json",
        "sha256": "71994379b81c65db111f652e747a00b599b5456c2a8242e9a0d6f379f560a14e",
        "license": "CC-BY-SA-4.0",
    },
    {
        "file": "hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc",
        "url": "https://lasp.colorado.edu/lisird/resources/lasp/hsrs/v2/hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc",
        "sha256": "84073a2b0fb5088fae75348f9204b8e6894dfccd16ef991303e545fcd966b84f",
        "license": "CC-BY-4.0",
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_source(path: Path, item: dict, verify_only: bool) -> bytes:
    if path.exists():
        data = path.read_bytes()
    elif verify_only:
        raise RuntimeError(f"Missing asset: {path}")
    else:
        print(f"Fetching official source: {item['url']}", flush=True)
        request = urllib.request.Request(item["url"], headers={"User-Agent": "M1-spectral-assets/1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        if digest(data) == item["sha256"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
    actual = digest(data)
    if actual != item["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual}; expected {item['sha256']}. "
                           "Refusing changed source data; do not substitute another spectrum.")
    return data


def canonical_tables(raw: Path) -> tuple[dict[str, bytes], dict]:
    cie = list(csv.reader(io.StringIO((raw / SOURCES[0]["file"]).read_text(encoding="utf-8"))))
    if len(cie) != 471:
        raise RuntimeError("CIE source must have 471 rows")
    for i, row in enumerate(cie):
        values = [float(v) for v in row]
        if len(values) != 4 or values[0] != 360 + i or not all(math.isfinite(v) and v >= 0 for v in values):
            raise RuntimeError("Invalid official CIE source row")
    cie_out = io.StringIO(newline="")
    writer = csv.writer(cie_out, lineterminator="\n")
    writer.writerow(["wavelength_nm", "x", "y", "z"])
    writer.writerows(cie)

    solar = list(csv.reader(io.StringIO((raw / "tsis1_hsrs_1nm.csv").read_text(encoding="utf-8"))))
    expected_header = ["wavelength (nm)", "irradiance (W/m^2/nm)",
                       "uncertainty (W/m^2/nm)", "bandwidth (nm)"]
    if solar[0] != expected_header or len(solar) != 25282:
        raise RuntimeError("Unsupported official TSIS-1 HSRS CSV schema")
    visible = []
    for i, row in enumerate(solar[1:]):
        values = [float(v) for v in row]
        if (len(values) != 4 or not all(math.isfinite(v) and v >= 0 for v in values)
                or abs(values[0] - (202.0 + 0.1 * i)) > 1e-7 or values[3] != 1.0):
            raise RuntimeError("Invalid TSIS-1 HSRS 1-nm-bandwidth source row")
        if 360.0 <= values[0] <= 830.0:
            # Copy the original decimal wavelength/irradiance strings. This is
            # a column/range selection, not a resampling or renormalization.
            visible.append(row[:2])
    if len(visible) != 4701:
        raise RuntimeError("TSIS-1 HSRS visible subset is incomplete")
    solar_out = io.StringIO(newline="")
    writer = csv.writer(solar_out, lineterminator="\n")
    writer.writerow(["wavelength_nm", "irradiance_w_m2_nm"])
    writer.writerows(visible)
    d65_all = list(csv.reader(io.StringIO((raw / "CIE_std_illum_D65.csv").read_text(encoding="utf-8"))))
    if len(d65_all) != 531:
        raise RuntimeError("CIE D65 source must cover 300..830 nm in 1 nm steps")
    for i, row in enumerate(d65_all):
        values = [float(v) for v in row]
        if len(values) != 2 or values[0] != 300 + i or not all(math.isfinite(v) and v >= 0 for v in values):
            raise RuntimeError("Invalid CIE D65 row")
    d65_visible = d65_all[60:]
    y = [float(row[2]) for row in cie]
    d65 = [float(row[1]) for row in d65_visible]
    integral_y = sum(0.5 * (a + b) for a, b in zip(y, y[1:]))
    d65_y = [a * b for a, b in zip(d65, y)]
    integral_d65_y = sum(0.5 * (a + b) for a, b in zip(d65_y, d65_y[1:]))
    factor = integral_y / integral_d65_y
    d65_out = io.StringIO(newline="")
    writer = csv.writer(d65_out, lineterminator="\n")
    writer.writerow(["wavelength_nm", "value"])
    writer.writerows((row[0], format(float(row[1]) * factor, ".17g")) for row in d65_visible)
    d65_metadata = {"doi": "10.25039/CIE.DS.hjfjmt59", "license": "CC-BY-SA-4.0",
                    "source_samples": 531, "samples": 471, "step_nm": 1.0,
                    "quantity": "relative spectral distribution", "normalization_range_nm": [360.0, 830.0],
                    "normalization": "trapz(D65_normalized*ybar)/trapz(ybar) = 1",
                    "normalization_quadrature": "trapezoidal rule, 1 nm grid, double precision",
                    "raw_y_integral_nm": integral_y, "raw_d65_y_integral_nm": integral_d65_y,
                    "multiplicative_normalization_factor": factor,
                    "interpolation": "linear", "extrapolation": "zero"}
    return {"cie_1931_2deg.csv": cie_out.getvalue().encode("ascii"),
            "solar_am0.csv": solar_out.getvalue().encode("ascii"),
            "d65_normalized.csv": d65_out.getvalue().encode("ascii")}, d65_metadata


def validate_model(path: Path) -> str:
    blob = path.read_bytes()
    resolution, nfine = 64, 288
    offset = 8 + 4 * resolution + 36 * resolution ** 3
    size = offset + 4 + 4 * (4 * nfine + 12)
    if (len(blob) != size or blob[:4] != b"SPEC" or struct.unpack_from("<I", blob, 4)[0] != resolution
            or struct.unpack_from("<I", blob, offset)[0] != nfine):
        raise RuntimeError("Unsupported RGB2Spec format; M1 requires official sRGB resolution 64 and forward block")
    data = array.array("f")
    data.frombytes(blob[8:offset] + blob[offset + 4:])
    if sys.byteorder != "little":
        data.byteswap()
    if not all(math.isfinite(v) for v in data):
        raise RuntimeError("Non-finite RGB2Spec coefficient")
    scale = data[:resolution]
    if scale[0] != 0.0 or scale[-1] != 1.0 or any(a >= b for a, b in zip(scale, scale[1:])):
        raise RuntimeError("Invalid RGB2Spec scale")
    matrix = struct.unpack_from("<9f", blob, offset + 4 + 16 * nfine)
    srgb = [.412453, .357580, .180423, .212671, .715160, .072169, .019334, .119193, .950227]
    if any(abs(a - b) > 1e-7 for a, b in zip(matrix, srgb)):
        raise RuntimeError("The coefficient file is not the supported sRGB model")
    return digest(blob)


def generate_model(third_party: Path, destination: Path) -> dict:
    if sys.byteorder != "little":
        raise RuntimeError("The pinned native-binary RGB2Spec generator needs a little-endian host for M1 assets")
    if shutil.which("cmake") is None:
        raise RuntimeError("CMake and a C++ compiler are required to rebuild the resolution-64 RGB2Spec model")
    with tempfile.TemporaryDirectory(prefix="m1-rgb2spec-build-") as temporary:
        build = Path(temporary)
        configure = ["cmake", "-S", str(third_party), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "rgb2spec_opt", "--parallel"], check=True)
        executables = [build / "rgb2spec_opt", build / "Release" / "rgb2spec_opt.exe", build / "rgb2spec_opt.exe"]
        executable = next((p for p in executables if p.is_file()), None)
        if executable is None:
            raise RuntimeError("CMake did not produce rgb2spec_opt")
        output = destination.with_suffix(".coeff.tmp")
        subprocess.run([str(executable), "64", str(output), "sRGB"], check=True)
        validate_model(output)
        output.replace(destination)
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
        compiler_lines = [line for line in cache.splitlines()
                          if line.startswith(("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS:", "CMAKE_CXX_FLAGS_RELEASE:"))]
    return {"method": "official rgb2spec_opt", "resolution": 64, "gamut": "sRGB",
            "arguments": ["64", "srgb.coeff", "sRGB"], "configuration": "Release",
            "compiler_cache": compiler_lines, "platform": platform.platform(),
            "byte_order": "little", "note": "Floating-point output may vary across compilers; actual SHA-256 is recorded."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="project root containing third_party/")
    parser.add_argument("--asset-dir", type=Path, default=Path("spectrum/m1"),
                        help="asset directory relative to --root (NeuralPFRenderer integration: spectrum/m1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="verify all checked-in assets; no network, compiler, or writes")
    mode.add_argument("--rebuild-model", action="store_true", help="explicitly regenerate the full resolution-64 sRGB table")
    args = parser.parse_args()
    root = args.root.resolve()
    assets = (root / args.asset_dir).resolve()
    raw = assets / "raw"
    vendor = root / "third_party" / "rgb2spec"
    if not args.verify_only:
        raw.mkdir(parents=True, exist_ok=True)
    upstream = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
    if upstream["commit"] != UPSTREAM_COMMIT or upstream["repository"] != "https://github.com/mitsuba-renderer/rgb2spec":
        raise RuntimeError("Unexpected RGB2Spec upstream manifest")
    for item in upstream["files"]:
        checked_source(vendor / item["path"], item, args.verify_only)
    for item in SOURCES:
        checked_source(raw / item["file"], item, args.verify_only)
    canonical, d65_metadata = canonical_tables(raw)
    for name, content in canonical.items():
        path = assets / name
        if args.verify_only:
            if not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(f"Canonical table differs from the real source: {path}")
        else:
            path.write_bytes(content)
    manifest_path = assets / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    model = assets / "srgb.coeff"
    generation = None
    if args.rebuild_model or (not model.exists() and not args.verify_only):
        generation = generate_model(vendor, model)
    model_hash = validate_model(model)
    if generation is None:
        if previous is not None:
            if model_hash != previous["files"]["srgb.coeff"]["sha256"]:
                raise RuntimeError("RGB2Spec coefficient checksum changed; use an explicit --rebuild-model for regeneration")
            generation = previous["rgb2spec"]["generation"]
        elif model_hash == BUNDLED_MODEL_SHA256:
            generation = {"method": "official rgb2spec_opt", "resolution": 64, "gamut": "sRGB",
                          "arguments": ["64", "srgb.coeff", "sRGB"],
                          "compiler": "g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
                          "flags": ["-O3", "-std=c++11", "-pthread"], "architecture": "x86_64",
                          "byte_order": "little", "note": "Generated from the pinned unmodified official source."}
        else:
            raise RuntimeError("Unrecorded RGB2Spec coefficients; use bundled assets or --rebuild-model")
    files = {}
    for path in [assets / name for name in canonical] + [model]:
        content = path.read_bytes()
        files[path.name] = {"sha256": digest(content), "bytes": len(content)}
    supporting_files = {}
    for relative in ["raw/TSIS_HSRS_v2_catalog_entry.json", "raw/TSIS_CSV_NETCDF_VERIFICATION.json",
                     "licenses/CIE-DATA-NOTICE.md", "licenses/TSIS-1-HSRS-NOTICE.md"]:
        path = assets / relative
        content = path.read_bytes()
        supporting_files[relative] = {"sha256": digest(content), "bytes": len(content)}
    manifest = {"schema": "m1-spectral-assets-v1", "prepared_date_utc": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                "wavelength_unit": "nm", "wavelength_min": 360.0, "wavelength_max": 830.0,
                "cie": {"source": "CIE 1931 2 degree, CIE 018:2019 Table 6",
                        "doi": "10.25039/CIE.DS.xvudnb9b", "samples": 471, "step_nm": 1.0,
                        "license": "CC-BY-SA-4.0", "interpolation": "linear", "extrapolation": "zero"},
                "d65": d65_metadata,
                "solar": {"source": "LASP LISIRD TSIS-1 HSRS, API tsis1_hsrs_1nm", "version": "2",
                          "quantity": "TOA spectral irradiance normal to solar center", "units": "W/m^2/nm",
                          "samples": 4701, "sampling_step_nm": 0.1, "spectral_bandwidth_nm": 1.0,
                          "transformation": "360..830 nm and first two columns selected verbatim; no resampling or normalization",
                          "interpolation": "linear", "license": "CC-BY-4.0",
                          "version_license_evidence": "raw/TSIS_HSRS_v2_catalog_entry.json",
                          "publication_doi": "10.1029/2022EA002637"},
                "sources": SOURCES,
                "rgb2spec": {"repository": upstream["repository"], "commit": UPSTREAM_COMMIT,
                             "license": "BSD-3-Clause", "generation": generation},
                "files": files, "supporting_files": supporting_files}
    if args.verify_only:
        if (previous is None or previous["schema"] != manifest["schema"] or previous["files"] != files
                or previous.get("sources") != SOURCES or previous.get("d65") != d65_metadata
                or previous.get("supporting_files") != supporting_files):
            raise RuntimeError("Missing or inconsistent spectral asset manifest")
        print("Verified: real CIE (471), real TOA solar (4701), normalized real D65 (471), official sRGB RGB2Spec (resolution 64).")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared spectral assets: {assets}")
        print(f"RGB2Spec SHA-256: {model_hash}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

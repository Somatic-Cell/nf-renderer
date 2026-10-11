#!/usr/bin/env python3
"""Compile every CMake-listed CUDA module and reject FP64 in the entire PTX.

Uses the real nvcc and SDK headers. It does not load an OptiX module, inspect
GPU SASS, link the application, or execute rendering. No fast-math option or
substitute headers are used. CPU reference code is outside this check.

The default run builds OptiX IR and PTX for the OptiX modules, and PTX for the
CUDA modules. Run from a Visual Studio Developer Command Prompt on Windows.
Example:
  python tools/validate_shader_fp32.py --optix-include "C:/OptiX/include"

--allow-fp64 is only for measuring an old baseline; the report still records
the failed FP32 condition. --ptx-only skips the additional IR build.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmake_modules(root: Path, variable: str) -> list[str]:
    """Read the first, literal source list; reject unsupported CMake syntax.

    This deliberately fails instead of silently omitting a target if the source
    lists stop being simple ${PROJECT_SOURCE_DIR}/... .cu paths.
    """
    source = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"\bset\s*\(\s*" + variable + r"\s+(.*?)\)",
                      source, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing CMake source list: {variable}")
    tokens = re.sub(r"#[^\n]*", "", match.group(1)).split()
    prefix = "${PROJECT_SOURCE_DIR}/"
    if not tokens or any(not p.startswith(prefix) or not p.endswith(".cu")
                         for p in tokens):
        raise ValueError(f"Unsupported source-list expression in {variable}")
    paths = [p[len(prefix):] for p in tokens]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate source in {variable}")
    for path in paths:
        if not (root / path).is_file():
            raise ValueError(f"Missing CUDA source: {path}")
    return paths


def source_snapshot(root: Path) -> dict[str, str]:
    """Conservative repository input set, including inactive code branches."""
    paths = [root / "CMakeLists.txt"]
    for directory in ("include", "shaders", "kernels", "utils"):
        paths.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix.lower() in
                     {".h", ".hpp", ".cuh", ".c", ".cpp", ".cu", ".inl"})
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def ptx_scan(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Preserve newlines when removing block comments, for PTX line references.
    clean = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"),
                   raw, flags=re.DOTALL)
    clean = re.sub(r"//[^\n]*", "", clean)
    files = {int(m.group(1)): m.group(2) for m in
             re.finditer(r'^\s*\.file\s+(\d+)\s+"([^"\n]+)"',
                         clean, re.MULTILINE)}
    symbols = {}
    symbol_pattern = re.compile(
        r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(",
        re.MULTILINE)
    for match in symbol_pattern.finditer(clean):
        symbols[clean.count("\n", 0, match.start()) + 1] = match.group(1)
    occurrences = []
    double_literals = []
    current_symbol = None
    current_location = None
    instruction_count = 0
    for number, line in enumerate(clean.splitlines(), start=1):
        if number in symbols:
            current_symbol = symbols[number]
            current_location = None
        location = re.match(r"\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if location:
            current_location = {
                "file": files.get(int(location.group(1)), location.group(1)),
                "line": int(location.group(2)),
                "column": int(location.group(3)),
            }
        text = line.strip()
        if re.search(r"\.f64\b", line):
            is_instruction = not text.startswith(".")
            instruction_count += int(is_instruction)
            occurrences.append({"ptx_line": number, "text": text,
                                "is_instruction": is_instruction,
                                "symbol": current_symbol,
                                "source_location": current_location})
        if re.search(r"\b0[dD][0-9a-fA-F]{16}\b", line):
            double_literals.append({"ptx_line": number, "text": text,
                                    "symbol": current_symbol,
                                    "source_location": current_location})
    return {
        "scope": "Entire module: all entries, functions, conversions and declarations; comments excluded",
        "symbols_seen": sorted(set(symbols.values())),
        "fp64_occurrence_count": len(occurrences),
        "fp64_instruction_line_count": instruction_count,
        "fp64_occurrences": occurrences,
        "fp64_literal_count": len(double_literals),
        "fp64_literals": double_literals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--optix-include", type=Path, required=True)
    parser.add_argument("--nanovdb-include", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cuda-cc", default="86")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--ptx-only", action="store_true")
    parser.add_argument("--allow-fp64", action="store_true")
    args = parser.parse_args()
    if not args.nvcc:
        parser.error("nvcc not found: set --nvcc or add the CUDA bin directory to PATH")
    if args.jobs < 1 or not re.fullmatch(r"[0-9]+[a-z]?", args.cuda_cc):
        parser.error("--jobs must be positive and --cuda-cc must be an architecture such as 86")
    root = args.root.resolve()
    optix = args.optix_include.resolve()
    nanovdb = (args.nanovdb_include or root / "ext/openvdb/nanovdb").resolve()
    for required in (root / "CMakeLists.txt", optix / "optix.h",
                     nanovdb / "nanovdb/NanoVDB.h"):
        if not required.is_file():
            parser.error(f"Required source/SDK file not found: {required}")
    output = (args.output or root / "build/shader_fp32_audit").resolve()
    report_path = (args.report or output / "shader_fp32_audit.json").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    optix_sources = cmake_modules(root, "OPTIX_KERNELS")
    cuda_sources = cmake_modules(root, "CUDA_KERNELS")
    if set(optix_sources) & set(cuda_sources):
        parser.error("A module is listed in both OptiX and CUDA source lists")
    jobs = [(source, "ptx") for source in optix_sources + cuda_sources]
    if not args.ptx_only:
        jobs += [(source, "optixir") for source in optix_sources]
    before = source_snapshot(root)
    version_result = subprocess.run([args.nvcc, "--version"], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if version_result.returncode:
        parser.error(f"nvcc --version failed: {version_result.stdout}")
    common = [args.nvcc, "--std=c++20", f"--gpu-architecture=compute_{args.cuda_cc}",
              f"--gpu-code=sm_{args.cuda_cc}", "--cudart", "none",
              "--relocatable-device-code=true", "--generate-line-info",
              "-I" + str(root / "include"), "-I" + str(root),
              "-I" + str(optix), "-I" + str(nanovdb)]
    if os.name == "nt":
        common.extend(["-Xcompiler=/utf-8", "-Xcompiler=/Zc:__cplusplus"])

    def compile_one(job: tuple[str, str]) -> dict:
        source, kind = job
        artifact = output / Path(source).with_suffix("." + kind)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        # Remove only this tool's exact output, so a failed compile cannot reuse
        # an artifact from an earlier invocation.
        if artifact.exists():
            artifact.unlink()
        command = [common[0], "--optix-ir" if kind == "optixir" else "--ptx",
                   *common[1:], source, "-o", str(artifact)]
        start = time.monotonic()
        result = subprocess.run(command, cwd=root, text=True, errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = artifact.with_suffix(artifact.suffix + ".log")
        log.write_text(result.stdout, encoding="utf-8")
        record = {"source": source, "kind": kind, "command": command,
                  "returncode": result.returncode,
                  "duration_seconds": round(time.monotonic() - start, 3),
                  "stdout": result.stdout, "artifact": str(artifact),
                  "log": str(log)}
        if result.returncode == 0 and artifact.is_file():
            record["artifact_sha256"] = sha256(artifact)
            record["artifact_bytes"] = artifact.stat().st_size
            if kind == "ptx":
                record["fp64_scan"] = ptx_scan(artifact)
        else:
            record["missing_artifact"] = not artifact.is_file()
        count = record.get("fp64_scan", {}).get("fp64_occurrence_count")
        print(f"{source} {kind}: rc={result.returncode}" +
              (f", FP64={count}" if count is not None else ""), flush=True)
        return record

    result_map = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(compile_one, job): job for job in jobs}
        for future in as_completed(pending):
            result_map[pending[future]] = future.result()
    results = [result_map[job] for job in jobs]
    after = source_snapshot(root)
    changed = [path for path in sorted(set(before) | set(after))
               if before.get(path) != after.get(path)]
    scans = [r["fp64_scan"] for r in results if "fp64_scan" in r]
    all_compiled = all(r["returncode"] == 0 and "artifact_sha256" in r for r in results)
    all_fp32 = (len(scans) == len(optix_sources) + len(cuda_sources) and
                all(s["fp64_occurrence_count"] == 0 and s["fp64_literal_count"] == 0
                    for s in scans))
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "reproduction_script": str(Path(__file__).resolve()),
        "reproduction_script_sha256": sha256(Path(__file__)),
        "compiler_version": version_result.stdout,
        "configuration": {"cuda_cc": args.cuda_cc, "cxx_standard": 20,
                          "relocatable_device_code": True, "line_info": True,
                          "fast_math": False, "ptx_only": args.ptx_only,
                          "allow_fp64_for_baseline": args.allow_fp64,
                          "optix_include": str(optix), "nanovdb_include": str(nanovdb)},
        "sdk_header_sha256": {str(p): sha256(p) for p in
                              (optix / "optix.h", nanovdb / "nanovdb/NanoVDB.h")},
        "source_lists": {"optix": optix_sources, "cuda": cuda_sources},
        "input_hash_scope": "All repository source/header files in include, shaders, kernels and utils, plus root CMakeLists.txt; includes inactive code",
        "input_sha256_before": before, "input_sha256_after": after,
        "inputs_changed_during_compile": changed,
        "compile_jobs": len(jobs), "ptx_modules_scanned": len(scans),
        "fp64_occurrence_count": sum(s["fp64_occurrence_count"] for s in scans),
        "fp64_instruction_line_count": sum(s["fp64_instruction_line_count"] for s in scans),
        "fp64_literal_count": sum(s["fp64_literal_count"] for s in scans),
        "all_compiled": all_compiled, "all_ptx_fp32": all_fp32,
        "passed": all_compiled and all_fp32 and not changed,
        "gpu_runtime_executed": False, "hardware_sass_inspected": False,
        "windows_application_compiled": False, "temporary_preinclude_or_stubs": False,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Compiled: {all_compiled}; all PTX FP32: {all_fp32}; changed inputs: {len(changed)}")
    return 0 if all_compiled and not changed and (all_fp32 or args.allow_fp64) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

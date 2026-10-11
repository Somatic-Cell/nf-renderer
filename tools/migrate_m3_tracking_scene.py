#!/usr/bin/env python3
"""Explicitly migrate one M3 numerical-reference JSON to the FP32 tracker.

The input is never overwritten. Physical parameters, cameras, materials and
render settings are preserved. Only the named atmosphere solver and its
numerical controls change, and every removed field is reported.
"""
import argparse
import json
import math
from pathlib import Path

OLD_CONTROLS = (
    "tauAbsoluteTolerance", "tauRelativeTolerance", "inverseTauTolerance",
    "distanceAbsoluteTolerance", "distanceRelativeTolerance", "maxQuadratureDepth",
    "maxQuadratureEvaluations", "maxInverseIterations",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratio-rr-threshold", type=float, default=0.01,
                        help="Unbiased ratio-product roulette threshold; 0 disables it (default: 0.01)")
    parser.add_argument("--max-tracking-events", type=int, default=1048576,
                        help="Candidate evaluation budget per segment (default: 1048576)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output file; the input and existing output are never overwritten")
    if not math.isfinite(args.ratio_rr_threshold) or not 0 <= args.ratio_rr_threshold <= 1:
        parser.error("--ratio-rr-threshold must lie in [0,1]")
    if not 1 <= args.max_tracking_events <= 4294967295:
        parser.error("--max-tracking-events must be a positive uint32")
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("rayleigh"), dict):
        parser.error("The input must contain a rayleigh object")
    settings = document["rayleigh"]
    old_solver = settings.get("solver", "numerical_reference")
    if old_solver not in ("numerical_reference", "delta_ratio_fp32"):
        parser.error("Only the numerical_reference or delta_ratio_fp32 solver can be migrated")
    removed = {key: settings.pop(key) for key in OLD_CONTROLS if key in settings}
    settings["solver"] = "delta_ratio_fp32"
    settings["ratioRrThreshold"] = args.ratio_rr_threshold
    settings["maxTrackingEvents"] = args.max_tracking_events
    # Validate/serialize before creating the output so an invalid JSON number
    # cannot leave a partially written scene behind.
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    print(json.dumps({"input": str(args.input), "output": str(args.output),
        "previous_solver": old_solver, "solver": settings["solver"],
        "removed_reference_controls": removed,
        "ratioRrThreshold": args.ratio_rr_threshold,
        "maxTrackingEvents": args.max_tracking_events}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compile real gpu_asset.cpp against TEST-ONLY host API doubles.

No CUDA/OptiX SDK or GPU is used. This tests upload memory bookkeeping,
not the inferencing-optimal layout transformation or neural inference.
Intended for GCC/Clang. Production builds MUST NOT include the mock directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='M4 source tree root')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--source', type=Path, help='Optional alternate gpu_asset.cpp')
    parser.add_argument('--expect-old-failure', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    root, here = args.root.resolve(), Path(__file__).resolve().parent
    source = (args.source or root / 'src/m4_nf/gpu_asset.cpp').resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = out / 'test_gpu_asset_host'
    command = [args.compiler, '-std=c++17', '-O1' if args.sanitize else '-O2',
               '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-fno-fast-math',
               '-I' + str(here / 'mock'), '-I' + str(root / 'include')]
    if args.expect_old_failure:
        command += ['-DM4_EXPECT_OLD_ALIGNMENT_FAILURE']
    if args.sanitize:
        command += ['-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    command += [str(here / 'test_gpu_asset_host.cpp'), str(source),
                str(root / 'src/m4_nf/asset.cpp'), '-o', str(exe)]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)
    subprocess.run([str(exe), str(root / 'assets/m4_nf/best/model.bin')], check=True)


if __name__ == '__main__':
    main()

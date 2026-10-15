# M4 NF validation record

## Actual environment and scope

- Linux x86-64; GCC 14.2.0; CMake 3.31.6; Python 3.13.5.
- PyTorch 2.10.0+cpu and NumPy. No CUDA toolkit, nvcc, OptiX SDK, or NVIDIA GPU.
- CPU implementation, exporter, numerical comparisons, and sanitizers were run.
- The original Zuko training package was **not** executed here. Numerical
  reference `equations` is an independent PyTorch replay of the fetched training
  source equations. A separate `--reference training` mode is provided to run
  the real training package and check its source hashes on the user's machine.
- No raw teacher CDF arrays were read or compared. The catalog supplies only
  the physical g/coefficient tables and provenance.
- GPU compilation, emitted-PTX checking, OptiX JIT, actual CoopVec evaluation,
  renderer integration, and frame-rate measurements are **not performed**.

## Executed checks

1. Release CMake build of the asset library and CPU probe. See `cpu_build.log`.
2. CTest: one executable containing six groups. See `core_tests.log`:
   binary16 finite-value round trips and rounding ties; domain rejection;
   1,000 oriented frames; 4,097 scalar RQS round trips with a conditioning-aware
   coordinate tolerance; 2,048 learned-NF sample/evaluate/mirror queries;
   anisotropic coefficient/proposal placement.
3. ASan+UBSan build and the same CTest executable. See `asan.log`.
4. 4,096 random fixed-direction PDF queries and 4,096 inverse samples, seed 731.
   See `cpu/report.json` and `parity.log`. Queries and native outputs are included
   as reproducible inputs for GPU testing. They are not teacher data.
5. Truncated asset, trailing bytes, and bad-magic rejection. See `asset_errors.json`.
6. Exporter run with checkpoint/snapshot/catalog integrity verification. A second
   independent export produced an identical `model.bin`; see `export_recheck.log`
   and `RESULTS.json`.

The `half_reference` backend rounds all MLP inputs/weights/biases/outputs to
binary16 but accumulates in sequential FP32 FMA on the CPU. It is a sensitivity
experiment, **not** a bit-exact Tensor Core or OptiX simulation. Actual device
arithmetic/order may differ. No GPU precision acceptance decision is inferred
from this experiment.

The real GPU probe compares GPU-FP32 and CoopVec using the same queries. Its
CMake target also compiles PTX and rejects `.f64` instructions/declarations and
double literals; that target remains unrun in this environment.

## Key results

- CPU FP32 fixed-direction log PDF maximum absolute error: 5.627083252e-5.
- CPU FP32 common-uniform maximum direction difference: 3.026062992e-6 rad.
- CPU FP32 sample/evaluate maximum log PDF disagreement: 8.345e-5.
- CPU half-sensitivity fixed-direction maximum log PDF error: 0.04000854262.
- CPU half-sensitivity common-uniform maximum direction difference:
  0.005727028206 rad (approximately 0.328 degrees).
- No invalid/nonfinite results among the 8,192 random queries in either backend.

These maxima describe only the recorded test set, not error bounds over the
whole condition/direction domain. Log PDF differences use the natural logarithm.
Round-trip errors depend on local conditioning and representability of the
returned FP32 direction. The reports include reference-side round-trip errors.

## Reproduction

From the package / renderer root:

```sh
cmake -S tests/m4_nf -B build-m4 -DM4_BUILD_OPTIX=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-m4 --parallel 2
ctest --test-dir build-m4 --output-on-failure
python tools/m4_nf/validate.py --asset assets/m4_nf/best/model.bin \
  --checkpoint PATH_TO_SNAPSHOT/best.pt --probe build-m4/m4_nf_probe_cpu \
  --reference equations --out output/m4_nf_equations
```

The Japanese root README contains the Visual Studio / OptiX commands and the
command using the unmodified training project as reference. Windows compilation
and the batch commands have not been executed here.

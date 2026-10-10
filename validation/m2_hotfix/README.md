# M2 invalid-path fix: validation

These are compiler/CPU validation records for the delivered source overlay.
They do not establish that the Windows application linked or ran on an NVIDIA GPU.
The user's original aggregate error did not identify its failure condition.

| Record | Scope |
|---|---|
| `m2_shader_compile.json` | Real nvcc commands, output, artifacts' hashes and all 43 repository shader/include hashes |
| `m2_shader_compile.log` | Full compiler output for 14 OptiX IR and 2 CUDA PTX modules; all succeeded |
| `cpu_tests.log`, `cpu_tests.xml` | Seven portable CTest checks; all passed |
| `day_tangent_audit.json` | Production tangent helper on 93,026 normal-mapped mixed-sign Day triangles, seven probes each |
| `audit_day_tangents.cpp` | Source of that audit, including material-binding and frame checks |
| `emitter_offset_repro.json` | Independent finite-triangle emitter/offset counterexample |
| `validation_metadata.json` | Repository/asset/dependency identity and explicit runtime limitations |

The Day audit used the production `m2/orca_scene.hpp`, `m2/orca_material.hpp` and
`m2/tangent_frame.hpp`, together with pinned Assimp. It inspected 651,182 probes;
all were valid and unit/orthogonality checks passed. All 279,078 probes where the
old scalar sign interpolation was exactly zero also passed. The 632 triangles
with exactly degenerate UVs produced 4,424 probes using the explicit inherited
Assimp chart. This is a sampled CPU geometry check, not a proof for every
barycentric coordinate or a GPU rendering test. No DDS texels were needed for
this frame-construction check; actual normal-map bindings were required.

The portable regressions can be reproduced from a normal repository checkout:

```sh
cmake -S tests/m1 -B build-checks -DCMAKE_BUILD_TYPE=Release
cmake --build build-checks --config Release
ctest --test-dir build-checks -C Release --output-on-failure
```

Initialize the existing `ext/json` submodule, or set `M1_JSON_INCLUDE_DIR` to the
directory containing `nlohmann/json.hpp`. The records here used GCC 13.3, CMake
3.31 and real CUDA 13.0.88/OptiX 9 headers. The nvcc command lines retain C++20,
compute_86/sm_86, RDC and line information, without fast math or API substitutes.
Existing unused-variable warnings remain in the compiler log.

`audit_day_tangents.cpp` is an additional asset audit, outside the seven portable
tests. Build it with C++17, this repository's `include/`, nlohmann/json and the
pinned Assimp include/library paths. It takes two positional arguments:
the official `EmeraldSquare_Day.fbx` path and its output JSON path.

The official FBX/DDS and compiled IR/PTX/executables are not included in this
overlay. The detailed implementation decisions and application steps are in
`docs/M2_INVALID_PATH_FIX_JA.md`.

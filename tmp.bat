python tools/m4_nf/run_gpu_validation.py ^
  --exe build-m4/Release/m4_nf_probe_gpu.exe ^
  --ir build-m4/m4_nf_probe.optixir ^
  --asset assets/m4_nf/best/model.bin ^
  --queries validation/m4_nf/cpu/queries.txt ^
  --cpu-reference validation/m4_nf/cpu/fp32.txt ^
  --out output/m4_nf_gpu
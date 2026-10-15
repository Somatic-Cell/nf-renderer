# M4: Conditional NF の FP32 / FP16 CoopVec 推論モジュール

基準 renderer: `Somatic-Cell/nf-renderer` commit `965a8f57202337cd4b8906c603dce4f2c46c32c8`
基準 training source: `Somatic-Cell/spherical-hg-flow` commit `1f9308bef5c53cfc59a93557d0d7101531eaba5f`

## 到達点と検証状況

今回は M4-A と M4-B に対応する独立した推論層である。
エクスポータ、生成済みモデル、FP32 の NF、FP16 の CoopVec MLP、OptiX 単体実行プログラムを含む。
M3 の描画ループへ雨滴を接続した完成版ではない。既存ファイルを上書きしない。

CPU の実ビルド・テスト・推論数値比較は実行済み。
この作業環境には nvcc、OptiX SDK、NVIDIA GPU がないため、GPU ソースのコンパイル、OptiX JIT、CoopVec 実行は未検証。
実機確認のためのプログラムを同梱している。GPU の結果は今回の CPU 検証記録には含まれない。
`half_reference` の結果は CPU の丸め感度試験であり、Tensor Core の演算を正確に再現した値ではない。

教師 CDF ファイルは読み込まず、学習モデルの教師に対する近似精度は評価していない。

## モデル

既定の同梱 asset は `best.pt` の **step 208000**。ZIP 名が218000でも最新 checkpoint と best は別である。
4 coupling layers、K=64、raw condition encoding、mirror symmetry。
各 MLP は `3 -> 64 -> 64 -> 64 -> {193,97}`、SiLU。
FP32 パラメータは72004個。構造やactivationを変更していない。
`assets/m4_nf/best/manifest.json` に checkpoint と asset の SHA-256、全テンソルの配置、物理データのfingerprintを記録した。

`model.bin` の内容は FP32 weights、91x300 の物理 g・散乱断面積・消散断面積、
波長/余弦の軸、HG の安定な FP32 条件準備に用いる194個の定数。
位相関数を角度 LUT に焼いたものではない。

## 配置

このフォルダーの内容を nf-renderer のルートへ同じ相対パスで追加する。
追加先は `include/m4_nf`, `src/m4_nf`, `shaders/m4_nf`, `tools/m4_nf`, `tests/m4_nf`,
`assets/m4_nf`, `validation/m4_nf`, `docs/M4_ANISOTROPIC_TRANSPORT_JA.md`。
既存の M1/M2/M3、build.bat、GUI、root CMakeLists は変更しない。

## 1. CPU 単体ビルド

PowerShell、renderer ルートで実行する。

```powershell
cmake -S tests/m4_nf -B build-m4 -DM4_BUILD_OPTIX=OFF
cmake --build build-m4 --config Release
ctest --test-dir build-m4 -C Release --output-on-failure
```

このビルドは Assimp、NanoVDB、DirectXTex を使わない。既存の `build` は削除しない。
既に `build-m4` を作成している場合、同じコマンドによる増分ビルドでよい。

## 2. OptiX 単体ビルド

`OPTIX_ROOT` は実際に使用中の SDK のインストール先へ置き換える。
CUDA は既存の nvcc を使用し、CoopVec のため OptiX 9 以上を要求する。

```powershell
cmake -S tests/m4_nf -B build-m4 `
  -DM4_BUILD_OPTIX=ON `
  -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
cmake --build build-m4 --config Release
```

CUDA が自動検出されない場合は、既存の CUDA Toolkit の場所を `-DCUDAToolkit_ROOT=...` で指定する。
MSVC C++ と直接起動する nvcc の両方に UTF-8 を指定済み。
`--use_fast_math` は指定しない。

生成物:

- `build-m4/Release/m4_nf_probe_gpu.exe` (Visual Studio generator)
- `build-m4/m4_nf_probe.optixir`
- `build-m4/m4_nf_probe.ptx`
- `build-m4/m4_nf_fp64_check.json`

PTX 内の `.f64` と倍精度リテラルを自動検査する。残った場合にはビルドを失敗させ、黙って許容しない。
この検査自身も、この環境では nvcc がないため未実行。

## 3. GPU 推論を比較する

同梱 `queries.txt` は4096件のPDF評価と4096件のサンプリング。教師ではなく、移植検証用の固定入力である。
比較対象の `fp32.txt` は同梱 CPU 推論で生成した出力。
NumPy を利用する Python で次を実行する。

```powershell
python tools/m4_nf/run_gpu_validation.py `
  --exe build-m4/Release/m4_nf_probe_gpu.exe `
  --ir build-m4/m4_nf_probe.optixir `
  --asset assets/m4_nf/best/model.bin `
  --queries validation/m4_nf/cpu/queries.txt `
  --cpu-reference validation/m4_nf/cpu/fp32.txt `
  --out output/m4_nf_gpu
```

GPU FP32 と実 CoopVec を同じ入力で実行する。warm-up と一回の計測 launch を分離する。
タイミングは推論だけの時間であり、Emerald のFPSではない。
結果は `output/m4_nf_gpu/report.json` と `fp32.log`, `coopvec.log`。
エラー時はそのログに残し、CPU/HGへの暗黙の代替はしない。

## 4. 学習側の実モデルとの比較

ローカル検証は、取得した学習ソースの数式を独立に再現した `reference_equations.py` を使っている。
学習側のプロジェクトそのものを動かしたという意味ではない。
学習環境が手元にある場合、次のコマンドは実際の `ConditionalSphereFlow` をimportして比較する。
`TRAINING_REPO` と `SNAPSHOT_DIR` の2パスを置き換える。教師データは不要。

```powershell
python tools/m4_nf/validate.py `
  --asset assets/m4_nf/best/model.bin `
  --checkpoint "SNAPSHOT_DIR/best.pt" `
  --probe build-m4/Release/m4_nf_probe_cpu.exe `
  --reference training `
  --training-repo "TRAINING_REPO" `
  --out output/m4_nf_training_parity
```

`conditional_model.py`, `hg.py`, `splines.py`, `sphere_splines.py` の SHA-256 を学習manifestと照合する。
違うソースを無断で比較対象にしない。

## 5. 最新stepの選択・再export

再exportには PyTorch (weights_only と safe_globals を利用) と NumPy 2 が必要。
既定は best.pt。step218000を明示的に使う場合:

```powershell
python tools/m4_nf/export_model.py `
  --snapshot "SNAPSHOT_DIR" `
  --catalog "CATALOG_FILE.json" `
  --selection checkpoint.pt `
  --out assets/m4_nf/step218000
```

既存の出力を黙って上書きしないため、空の出力ディレクトリを指定する。
FP16変換と GPU 最適行列配置への変換は `GpuAsset::upload()` が実行時に行う。
行列積の隠れ層を減らす、activation をReLUに替える、といった変更は行わない。

## 数値仕様

### MLP

FP32 backend は row-major の FP32 parameters を FMA で評価する。
CoopVec backend は初段入力を3から16、最後の出力を193から208/97から112へ0パディングし、
推論最適レイアウトへ変換する。数学的な出力次元は元の193/97のまま。
線形層はFP16、各隠れ層のSiLUはFP32、その結果を次のFP16入力に丸める。
RQSのlogitsは出力のhalfからfloatへ昇格してから使用する。

### HG と RQS

HG の端点距離表現、区間と円周の RQS、全64bin、mirror symmetry、境界微分の結合、
正負の入射余弦、順変換/逆変換の順序を維持する。
サンプリング時のinverseは反復や近似逆関数でなく、学習コードと同じ解析式。
物理準備の HG knot masses だけは FP64 CDF の差ではなく、代数的に等価な質量式でFP32計算する:

`mass = (1-g)*(1+g)*delta_mu / (s0*s1*(s0+s1))`。

ここにdensity floorは追加しない。元モデルのmin-bin/derivative制約は維持する。
累積和の丸め順序は PyTorch と bitwise 一致を主張せず、数値比較結果を記録する。

### 条件表の端点

NNはc=[-1,1]を許容するが、外部のg表は約±0.999986292まで。
現モジュールはその外を `condition_out_of_domain` として拒否する。
全方向の媒質レンダリングに接続する前に、endcapの扱いを決める必要がある。
現段階でclamp・線形外挿・HG fallbackを無断で加えていない。

## 異方性 RTE

`docs/M4_ANISOTROPIC_TRANSPORT_JA.md` が正式な接続規約。
`anisotropic.hpp` の関数は、方向依存の断面積、単一雨滴の位相関数、物理 kernel、
Jakob の出射側係数を外に出す定式化、camera proposalのPDFを分離する。
`cross_sections()` は MLP・RQS を呼ばず、delta/ratio tracking に使用できる。
まだ雨域の輸送ループは変更していない。

# M4 CoopVec network stride 修正

## 対象と変更範囲

`nf-renderer-M4-NF-inference-965a8f5.zip` と前回の
`nf-renderer-M4-NOMINMAX-fix.zip` を適用済みの M4 推論単体プログラムが対象です。
製品コードの変更は **`src/m4_nf/gpu_asset.cpp` 1 ファイルだけ**です。

今回の原因は NN の FP32 演算ではなく、推論開始前の GPU アセット準備です。
`probe_optix.cpp` は FP32/CoopVec どちらの mode でも同じ `GpuAsset::upload()` を
呼び出すため、先に実行される FP32 の検証中にも CoopVec の行列配置変換が走ります。
この mode 間の共通初期化は今回は変更していません。

## ログと原因

```text
outputNetworkStrideInBytes (0x26BE0) must be 64 byte aligned.
M4 NF PROBE ERROR: OptiX: OPTIX_ERROR_INVALID_VALUE
```

行列・バイアスの各開始オフセットは 128 byte 境界にそろえていましたが、
最後のバイアスを加算した**ネットワーク全体の末尾**をそろえていませんでした。
最後のバイアスは、論理出力 97 を 112 へゼロパディングした FP16 配列で
`112 * 2 = 224 byte` です。128 byte 境界から 224 byte 進むため、
その末尾は 64 で割って 32 余ります。

ログの `0x26BE0 = 158688` も `158688 % 64 = 32` です。
既存の 128 byte 配置規約で切り上げると `0x26C00 = 158720` になります。
この例では末尾の未使用領域が 32 byte 増えるだけです。

OptiX SDK 9.0 のヘッダーには単一ネットワークでは stride が無視される旨の
コメントがありますが、今回の実行環境では `numNetworks=1` でも
上記の整列検査が行われています。stride を 0 にして検査を迂回せず、
有効な正の整列済み stride と、それ以上の実際の確保容量を渡します。

## 修正内容

- 入力・出力ネットワークの全体 stride を `align128()` で切り上げる。
- `cuMemAlloc()` と `optixCoopVecMatrixConvert()` に同じ整列済み値を渡す。
- 入力 raw buffer の末尾の追加領域はゼロで初期化する。
- 各行列・バイアスのオフセット、要素数、値、精度を変更しない。
- 行列オフセットの 64 byte 整列、バイアスの 16 byte 整列、
  および各領域が確保範囲内に収まることを変換前に検査する。
- 整列計算の整数オーバーフローを検査する。
- 非同期の行列変換の完了を明示的に待ってから、同じバッファの
  バイアス領域へ同期 HtoD コピーする。

モデル本体・重みの export・RQS・HG・方向生成・PDF・混合精度設定・
CUDA シェーダー・M3 の描画処理・CMake・LaunchParams は変更しません。
前回の `numeric_limits` への NOMINMAX 対策も保持しています。

## 適用

ZIP の `src/m4_nf/gpu_asset.cpp` をリポジトリの同じ相対パスへ上書きしてください。
手元の編集を保持する場合は、同梱差分を確認して取り込んでください。
上書きとパッチ適用はどちらか一方を行います。

```powershell
# パッチをリポジトリ直下に置いた場合
 git apply --check m4-coopvec-stride-fix.patch
 git apply m4-coopvec-stride-fix.patch
```

## 再ビルド

既存の構成済み `build-m4` を使い、リポジトリ直下で実行します。

```powershell
cmake --build build-m4 --config Release --target m4_nf_probe_gpu
```

今回の更新のために clean、build ディレクトリの削除、重みの再 export は不要です。
シェーダーを変更していないため、当該 CPP の再コンパイルと実行ファイルの
再リンクが対象です。既存の PTX 検査ターゲットは通常どおり走ります。

再び同じ比較コマンドを実行します。

```powershell
python tools/m4_nf/run_gpu_validation.py `
  --exe build-m4/Release/m4_nf_probe_gpu.exe `
  --ir build-m4/m4_nf_probe.optixir `
  --asset assets/m4_nf/best/model.bin `
  --queries validation/m4_nf/cpu/queries.txt `
  --cpu-reference validation/m4_nf/cpu/fp32.txt `
  --out output/m4_nf_gpu
```

成功すれば `report.json`、`fp32.log`、`coopvec.log` が更新されます。
別の理由で止まった場合は、エラー表示に指定された当該実行のログを確認します。

## 今回実施した検証と限界

1. GCC と Clang で、実際の `gpu_asset.cpp` を **テスト用ホスト API モック**へ接続。
   変更前にはログと同じ `0x26BE0` の stride と拒否条件を再現し、
   変更後は `0x26C00` で通過しました。
2. SDK が返す行列容量が異なることを想定した 2 種類の合成サイズで、
   16 行列・16 バイアスの範囲と配置を検査しました。
   この合成サイズは実デバイスの optimal-layout 容量の保証ではありません。
3. 同梱の学習済み `model.bin` を読み込み、転送した全 FP32 フィールドと
   パディング後の全 FP16 行列・バイアスをバイト単位で検査しました。
4. 再 upload、二重 release、意図的な変換エラー時の解放を検査しました。
5. 同じホスト API モック試験を ASan / UBSan で実行し、成功しました。
6. M4 の既存 CPU CMake プロジェクトを Release ビルドし、CTest
   1 実行ファイル内の既存 6 群がすべて成功しました。

**CUDA コンパイラ・GPU・OptiX 実行環境は今回の検証環境にありません。
Windows / MSVC の GPU probe ビルド、OptiX の実変換・JIT、GPU 推論は未実施です。**
ホスト API モックは配置・確保・転送の検査専用で、実際の optimal-layout
変換や CoopVec の数値計算を再現するものではありません。

`validation/m4_coopvec_stride/mock` はテスト専用です。
**レンダラや GPU probe の include path に加えないでください。**
通常の適用で必要なのは製品コード 1 ファイルだけです。

GCC/Clang 環境でのホスト API モックの再実行例:

```bash
python validation/m4_coopvec_stride/run_host_mock.py --root . --out build-m4-host-layout
python validation/m4_coopvec_stride/run_host_mock.py --root . --out build-m4-host-layout-asan --sanitize
```

## 参照

- ユーザーの `fp32(1).log` に記録された実行時診断。
- OptiX SDK 9.0 固定タグの `optix_host.h`:
  `https://raw.githubusercontent.com/NVIDIA/optix-dev/v9.0.0/include/optix_host.h`
- 行列・バイアスのオフセットの整列要件:
  `https://raytracing-docs.nvidia.com/optix9/api/group__optix__device__api__coop__vec.html`


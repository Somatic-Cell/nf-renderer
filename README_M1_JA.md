# NeuralPFRenderer — M1 変更・追加ファイル

**既存の NeuralPFRenderer に、単一波長のスペクトラル表面パストレーサーを追加する差分です。**
既存の `Renderer`、`CUDABuffer`、Assimp／DirectXTex、OptiX pipeline／SBT、カメラ、GUI を使用します。
今回の指定に従い `cuda_driver_wrapper` は導入しません。M0 のファイルは適用の前提にしていません。

基準は公開リポジトリ
[Somatic-Cell/NeuralPFRenderer](https://github.com/Somatic-Cell/NeuralPFRenderer) の
commit **`242e8530b6f565cd7bc5d839bc7569fa3128b477`** です。
コードの作成日・検証日は 2026-10-10。配布 ZIP は変更／追加ファイルの overlay であり、
既存リポジトリや外部 submodule 一式を含む clone ではありません。

**CPU 数値検証と実 nvcc による全 16 translation unit のコンパイルは通過しています。
Windows アプリ全体のビルド、OptiX の実 GPU 起動、画像比較は未実施です。**
したがって、このファイル一式を「M1 の GPU 受入試験まで完了したもの」とは扱いません。
検証可能なソース、実データ、小型シーン、実施済み検証のログをまとめたものです。

## 1. 今回の範囲

| 項目 | M1 の内容 |
|---|---|
| 波長 | 1 パスにつき 1 波長、360–830 nm の一様 sampling。途中で変更しない |
| スペクトルフィルム | scalar radiance／throughput。波長 PDF の補償は一度だけ。CIE XYZ から linear sRGB へ変換 |
| ポリゴン | 既存 Assimp loader、GAS／IAS／SBT、instance transform を使用 |
| RGB texture | CUDA による sRGB 復号後、linear RGB をフィルタしてから公式 RGB2Spec へ入力 |
| BSDF | Lambert、既存の Lambert + GGX metallic mixture、分散を持つ理想ガラス |
| 法線 | 頂点法線、既存 normal-map 規約、独立した normal UV。Ns／Ng の役割を分離 |
| 太陽 | 有限円盤、実 TOA 太陽 SPD、表面からの NEE、BSDF continuation からの hit |
| 面光源 | 三角形の面積 sampling、光源自身の world transform、emitter-hit の逆 PDF |
| MIS | power heuristic。`mis`、`bsdf`、`nee` の比較モード |
| Alpha | primary／shadow とも stochastic coverage。光源自身の alpha も反映 |
| 出力 | 既存 PNG に加え、無クランプ・トーンマップ前の linear RGB PFM と設定 JSON |
| 大気・雨滴 | M1 は真空。Rayleigh／球面大気／多重散乱／虹／rainbow の重み読込は後続工程 |

M1 では medium の透過率は全区間で 1 です。以前の工程分けでは、実 Emerald の
importer と画像受入が M2、Rayleigh 大気が M3 に当たります。
今回の小型シーンを Emerald の完成画像として代用していません。

既存の RGB と旧 spectral の entry は残し、`integrator.m1Surface = true` のときに
既存 spectral entry から M1 に分岐します。共通部分の不具合修正には、texture の
初期化順、二重 sRGB 復号、面光源の座標変換、コンパイル阻害箇所などを含みます。
これらの修正は従来モードにも影響します。

## 2. ZIP の使い方

ZIP を既存リポジトリとは別のフォルダへ展開してください。ZIP 内は次の構成です。

| パス | 用途 |
|---|---|
| `files/` | リポジトリのルートからの相対パスで並べた、変更／追加ファイルの本体 |
| `manifest.json` | 基準 commit、各ファイルの適用前／適用後 SHA-256 |
| `apply_m1.py` | Python 標準ライブラリだけで動く適用スクリプト |
| `existing-files.patch` | 既存ファイルへの変更をレビューするための unified diff。追加ファイルは `files/` に収録 |
| `CHANGED_FILES.md` | 修正と追加の全ファイル一覧 |

展開先の PowerShell から、まず確認します。

```powershell
python .\apply_m1.py --repo C:\work\NeuralPFRenderer --check
```

問題がなければ適用します。

```powershell
python .\apply_m1.py --repo C:\work\NeuralPFRenderer --apply
```

`--apply` がなければ書き込みません。変更対象の既存ファイルが基準と異なる場合は、
適用全体を拒否します。UTF-8 BOM と CRLF の違いだけは許容します。
既存ファイルのバックアップを作り、すでに同じ内容が適用済みならスキップします。
競合が出たファイルは `existing-files.patch` と `files/` を参照して、手元の変更と
統合してください。未コミットのコードを強制上書きするオプションは設けていません。

M0 の ZIP を探したり、先に M0 をコピーしたりする必要はありません。

## 3. 主要な変更ファイル

| ファイル群 | 変更の目的 |
|---|---|
| `include/sceneDescIO.hpp`、`src/main.cpp` | M1 設定の JSON 入出力。明示されたシーンの欠落／構文エラーを失敗にする |
| `include/launch_params.h`、`include/m1/render_params.h` | M1 設定と実スペクトルデータの GPU pointer を既存 launch params に追加 |
| `include/renderer.hpp`、`src/renderer.cpp` | 既存資源経路での upload、面光源表、alpha traversal、OptiX stack 計算、診断カウンタ |
| `shaders/device/m1_surface.cuh` | M1 の表面 transport、太陽／面光源 NEE、MIS、RR、単一波長 film |
| `shaders/entry/{raygen,ch_radiance,miss_radiance}.cu` | 既存 entry 名のまま M1 へ分岐 |
| `shaders/entry/{ah_radiance,ah_shadow}.cu` | alpha の整合と M1 shadow における glass の扱い |
| `shaders/callable/light_sample.cu` | 面光源の world-space geometry を受光面の transform で再変換しない |
| `include/m1/bsdf.hpp`、`math.hpp` | CPU／CUDA 共通の BSDF 評価・sampling・PDF |
| `include/m1/{spectral_types,spectral_assets,rgb2spec_device}.hpp`、`src/m1/spectral_assets.cpp` | 実データ検査・読込・公式 RGB2Spec の GPU lookup |
| `include/model.h`、`src/model.cpp`、`src/texture.cpp` | 初期値、emission texture、sRGB encoding、既存 importer の接続修正 |
| `include/application.hpp`、`src/application.cpp`、`kernels/toneMap.cu` | 既存 GUI と PFM 保存。表示段階だけで負の RGB をクリップ |
| `CMakeLists.txt`、既存の関連 helper headers／kernels | 追加ソースのビルドと、実コンパイルで検出した阻害箇所の最小修正 |

詳細なモデル・測度・近似の説明は、
[`docs/M1_INTEGRATOR.md`](docs/M1_INTEGRATOR.md)、
[`docs/M1_SPECTRAL_DATA.md`](docs/M1_SPECTRAL_DATA.md)、
[`docs/M1_HOST_INTEGRATION.md`](docs/M1_HOST_INTEGRATION.md) にあります。

## 4. 実データは同梱済み

`spectrum/m1/` は、そのまま読み込める完成したデータ一式です。

| データ | 固定した内容 |
|---|---|
| CIE | 公式 CIE 1931 2°、360–830 nm、1 nm 間隔、471 点 |
| 太陽 | LASP TSIS-1 HSRS Version 2、1 nm 帯域幅／0.1 nm 間隔、4,701 点。W m⁻² nm⁻¹ |
| D65 | 公式 CIE D65。固定の 360–830 nm 区間で Y 重み付き平均が 1 となるよう正規化 |
| RGB2Spec | 公式 commit `721145dedf2491851bd46ab8fd165955cb38ddaf`、sRGB、resolution 64 の実生成モデル |

太陽の入力は法線面の放射照度です。角半径を `a` とすると、円盤の radiance には
`E_lambda / (pi * sin(a)^2)`、sampling PDF には `1 / (4*pi*sin(a/2)^2)` を使います。

RGB 発光は `s = 2 * max(R,G,B)` とし、
`lightIntensityFactor * s * RGB2Spec(RGB/s, lambda) * normalized_D65(lambda)`
というモデルで定義します。RGB から実物の SPD が一意に復元できるとは仮定しません。
emissive texture は emission 定数に乗算します。reflectance の texture は既存の
規約どおり、定数 base color を置き換えます。

`spectrum/m1/manifest.json` に出典 URL・単位・ハッシュ・生成条件、`raw/` に原資料、
`licenses/` と `third_party/rgb2spec/LICENSE` に許諾表示を含めています。
再配布の際はこれらを保持してください。CIE／D65 は CC BY-SA 4.0、TSIS は
CC BY 4.0、RGB2Spec の実装は BSD-3-Clause です。詳細な根拠は同梱資料を参照してください。

通常のビルド・起動で optimizer の実行やデータ取得は必要ありません。
Python による同梱データの破損確認だけなら、ネットワーク接続も不要です。

```powershell
python tools\prepare_spectral_assets.py --verify-only
```

## 5. ビルドと起動

既存プロジェクトの Windows／Visual Studio 2022／CUDA／OptiX 9 と外部 submodule を
使用します。今回別の CUDA 資源ラッパーは追加していません。
以下は既存依存関係をインストール済みの場合の例です。SDK と vcpkg の場所は
手元の環境に合わせてください。

```powershell
cd C:\work\NeuralPFRenderer
git submodule update --init --recursive

cmake -S . -B build-m1 -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  "-DOPTIX9_INSTALL_DIR=C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"

cmake --build build-m1 --config Release
.\build-m1\bin\Release\NeuralPFRenderer.exe m1_area_mis.json
```

作業ディレクトリはリポジトリのルートにしてください。`m1AssetDirectory` はこの
ディレクトリを基準に解決します。シーンは明示パスまたは `scene/`、モデルは明示パス
または既存の `model/mesh/` から解決します。`build.bat` は既存 `build` を削除するので、
ここでは上記の CMake コマンドを使い、`build-m1` に出力しています。

`CUDA_CC` は従来値の 86 を cache 変数にしたため、必要なら configure 時に変更できます。
shader module と host は同じヘッダ・同じ設定で再ビルドしてください。古い `.optixir`
や `.ptx` を混ぜると launch params／SBT の型が一致しません。

Windows の full build はこの環境では実行していません。実施済みの shader compile
環境は CUDA 13.0.88、OptiX 9、C++20、compute_86／sm_86 です。

## 6. 検証用シーン

`model/mesh/m1/` は、この差分用に作成した小型 courtyard、閉じた glass box、
Lambert、rough metallic／plastic、RGB checker、alpha card、独立した 2 個の面光源です。
メートル単位、Y-up。面光源に移動・回転・非一様 scale を与え、受光面と別の
instance transform を使う経路を通します。NVIDIA の Emerald asset は含めていません。

| 照明 | MIS | BSDF only | NEE only |
|---|---|---|---|
| 面光源のみ | `m1_area_mis.json` | `m1_area_bsdf.json` | `m1_area_nee.json` |
| 太陽のみ | `m1_sun_mis.json` | `m1_sun_bsdf.json` | `m1_sun_nee.json` |
| 両方 | `m1_combined_mis.json` | `m1_combined_bsdf.json` | `m1_combined_nee.json` |

`NEE only` でも camera ray と delta event 直後の emitter-hit は保持します。
glass を通る屈折経路を無条件に消す設定ではありません。太陽は小さいため
BSDF-only の分散が大きくなります。まず面光源のみの比較から始めてください。

既存 loader に合わせ、fixture の `map_Ks` は従来の packed texture として
R=AO、G=roughness、B=metallic を格納しています。一般的な Phong specular map と
同じ意味ではありません。Emerald の材質 workflow は M2 で実 asset を監査します。

各 JSON の主要な追加設定は次です。

```json
{
  "environment": {"file": ""},
  "integrator": {
    "applySpectralRendering": true,
    "m1Surface": true,
    "m1DirectMode": "mis",
    "spp": 4,
    "maxBounce": 12,
    "isAccumulate": true,
    "m1RrDepth": 5,
    "m1Seed": 17,
    "m1SunEnabled": true,
    "m1SunScale": 1.0,
    "m1SunAngularRadiusDeg": 0.25,
    "m1SunZenithDeg": 35.0,
    "m1SunAzimuthDeg": 65.0,
    "m1RayEpsilon": 0.0001
  },
  "spectrum": {"m1AssetDirectory": "spectrum/m1"},
  "enableGroundRendering": false
}
```

M1 で volume、HDR environment、従来の巨大 ground sphere を指定すると明示的に
エラーにします。大気が描けたように見せるために従来の sky LUT を読む処理はありません。

## 7. PNG と linear PFM

GUI の **Save PNG + linear PFM** から、同じ保存操作で以下を出力します。

- `output/<name>.png`：現在の表示バッファ、従来の exposure／tonemap／gamma 後。
- `output/<name>.linear.pfm`：常に radiance color film の float32 linear RGB。
- `output/<name>.linear.json`：total spp、seed、mode、depth、太陽、カメラなどの実設定。

PFM の値は CIE XYZ の生の波長積分を linear sRGB に変換したものです。Y 積分での
除算や 683 の係数を含みません。数値比較には PNG ではなく PFM を使ってください。
表示が明るい場合は Exposure を調整します。これは PFM の数値を変更しません。
PFM は bottom-to-top、scale の符号で endian を記録します。

```powershell
python tools\m1_pfm_stats.py output\area_mis_COLOR.linear.pfm output\area_nee_COLOR.linear.pfm
```

`--roi X Y WIDTH HEIGHT` を追加すると、左上原点の範囲を集計できます。
これは保存画像の統計を読むツールであり、未実施の画像比較を成功にするものではありません。

## 8. 実施済みの検証と残る受入

| 検証 | 結果 |
|---|---|
| C++17 の CPU CMake build／CTest | 3 実行ファイル、全件通過 |
| BSDF 数値検証 | 9 群通過。独立積分、解析値、Fresnel／TIR、GGX、RR／MIS、Ns／Ng support |
| RGB2Spec 公式実装との比較 | 10,126 色。評価値の最大絶対差 `7.56979e-6` |
| 実 CIE／D65 を使う色再現、SPD の単位・範囲・順序、破損入力 | 通過 |
| シーン | 9 個の M1 JSON と 2 個の既存 JSON を読込。明示入力のエラーが元ファイルを変更しないことを確認 |
| 実 nvcc | 14 OptiXIR + 2 PTX、全 16 translation unit 成功 |
| 共有 POD、OptiX stack helper、既存 CoopVec host headers | 実 CUDA／OptiX headers で構文・型のコンパイルを確認 |
| Windows アプリ全体 | 未検証。Linux 上の full host compile 試行は DirectXTex.h 不在で停止 |
| GPU の起動・描画・速度 | 未実施 |
| Emerald の実 asset と画像 | 未実施 |

`validation/` に実行コマンド、ログ、ソースの SHA-256 を残しています。
配布はソースとデータを目的とし、検証時の一時 object／実行ファイルは含めません。

CPU の検証だけを再実行する場合は、アプリ全体とは独立に次を実行できます。

```powershell
cmake -S tests/m1 -B build-m1-cpu
cmake --build build-m1-cpu --config Release
ctest --test-dir build-m1-cpu -C Release --output-on-failure
python tools\prepare_spectral_assets.py --verify-only
```

CPU 検証も既存 `ext/json` の header を使います。別の配置の場合は
`-DM1_JSON_INCLUDE_DIR=<nlohmann/json.hpp を含むディレクトリ>` を指定できます。

GPU を使う残りの受入では、まず小型シーンの起動と texture／alpha／光源の向きを確認し、
同じ camera・depth・SPD で BSDF／NEE／MIS の PFM を比較します。独立 seed の反復から
平均と誤差を評価し、画像中の画素間の違いをそのまま Monte Carlo の標準誤差と
みなさないでください。tag 付き／なしの sRGB texture が一致すること、glass による
shadow、normal map、非一様 transform、蓄積 reset も実 GPU での確認対象です。

## 9. 後続の大気・ネットワーク実装へ

今後 Rayleigh を足す位置は、`m1_surface.cuh` の surface trace 前に行う medium
free-flight と、太陽 shadow connection の transmittance です。M1 の spectral film、
表面 BSDF、geometry、light sampling を利用できます。

rainbow／spherical-hg-flow の network 構成・condition・weight layout・forward／inverse
の契約は、この差分では確定していません。旧 NF の重みを新モデルの形式と仮定して
読み込む処理は入れていません。M1 は起動時に旧 Mie data／NF weight／atmosphere LUT
のロードを省き、network が未提供でも表面側の実装を検証できる構成にしています。

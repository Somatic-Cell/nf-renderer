# M1: 実スペクトル資産と RGB2Spec の組み込み

この実装は、既存の NeuralPFRenderer に組み込むための実データ、読み込み処理、
RGB2Spec の CPU/GPU 共通ルックアップを提供する。CIE、太陽、D65 のデータは
各機関の公式配布物であり、ガウス関数や黒体による代用品ではない。
RGB2Spec の係数は、固定した公式実装から **sRGB、解像度 64** で実際に生成した。

## 1. 同梱内容

以下の表はリポジトリのルートからのパス。`spectrum/m1/` に同梱したデータを、
既存 Renderer の `uploadSpectrumData()` が `load_spectral_assets()` に渡す。

| ファイル | 内容・単位 | レンダラでの用途 |
|---|---|---|
| `spectrum/m1/cie_1931_2deg.csv` | CIE 1931 2° 等色関数、360–830 nm、1 nm 間隔、471 点 | 1 波長の放射輝度を XYZ の寄与に変換 |
| `spectrum/m1/solar_am0.csv` | TSIS-1 HSRS v2、法線面分光放射照度 W m⁻² nm⁻¹、360–830 nm、0.1 nm 間隔、4,701 点 | 有限太陽円盤の放射輝度 |
| `spectrum/m1/d65_normalized.csv` | 実 CIE D65 を後述の規則で正規化、360–830 nm、1 nm 間隔 | 既存 RGB 面光源の発光スペクトルを定義 |
| `spectrum/m1/srgb.coeff` | 公式 RGB2Spec の sRGB 係数テーブル、解像度 64、9,442,108 bytes | linear sRGB → 反射率スペクトル |
| `spectrum/m1/raw/` | 取得した CSV・CIE metadata・TSIS NetCDF・版と許諾の根拠 | 元データの保存・追跡 |
| `spectrum/m1/manifest.json` | URL、SHA-256、単位、規格化、生成条件、係数の実ハッシュ | 再現性・破損検出 |
| `spectrum/m1/licenses/` | CIE と LASP の出典・許諾・変更内容 | データ再配布時の表示 |
| `third_party/rgb2spec/` | 公式ソースと BSD-3-Clause LICENSE、UPSTREAM.json | ホスト実装と係数の再生成 |

### CIE 1931 2°

公式 CIE データセット DOI は `10.25039/CIE.DS.xvudnb9b`。
原典は CIE 018:2019 の Table 6。数値は元 CSV の文字列を保ち、
ヘッダ追加と改行コードの変更だけを行っている。元ファイルは別途そのまま保存した。
ライセンスは元 metadata が指定する **CC BY-SA 4.0**。出典と許諾は
[CIE の公式ページ](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer)
および同梱 metadata を参照する。

### 太陽の実 SPD

採用したのは **LASP TSIS-1 Hybrid Solar Reference Spectrum Version 2** の
1 nm 帯域幅版である。「1 nm」は分光分解能であり、サンプリング間隔は
**0.1 nm**。API CSV の全域は 202–2730 nm、25,281 点。
そこから 360–830 nm と最初の 2 列を選び、数値の文字列をそのまま保持した。
補間、再正規化、地上への透過率の乗算は準備段階で行っていない。

この値は太陽距離 1 AU における大気上端の分光放射照度で、太陽活動極小期の
基準スペクトルである。毎日の太陽変動を表すものではない。公式 v2 NetCDF の
`SSI` と API CSV の 25,281 点を照合し、放射照度は float64 で全点完全一致した。
波長だけは CSV の十進表記により最大 `4.55e-13 nm` の差がある。
確認結果は `raw/TSIS_CSV_NETCDF_VERIFICATION.json` に保存した。

版と **CC BY 4.0** の許諾は、論文のライセンスから推測せず、
[公式 LISIRD catalog](https://lasp.colorado.edu/lisird/latis/catalog) の dataset entry で確認した。
その entry が 1 nm 版の v2 NetCDF を明示している。対応する一次資料は
[Coddington et al. (2023)](https://doi.org/10.1029/2022EA002637) と
[LASP のデータページ](https://lasp.colorado.edu/lisird/data/tsis1_hsrs/)。

参考として、採用した可視域 360–830 nm の台形積分は約
`739.3906237986226 W m^-2` である。この値を全波長の太陽定数と取り違えない。

## 2. スペクトルと測度の契約

### 1 path につき 1 wavelength

波長の単位は nm。波長サンプルの密度 `q_lambda` の単位も nm⁻¹ とし、
パスで得た `L_lambda` に対し、フィルムに

\[
  \Delta(X,Y,Z)=\frac{L_\lambda}{q_\lambda}
  (\overline{x}(\lambda),\overline{y}(\lambda),\overline{z}(\lambda))
\]

を加算する。たとえば `[360,830)` の一様サンプリングなら
`q_lambda = 1/470 nm^-1`。標本数で平均を取る処理はこれとは別。
この段階では 683 を乗じず、Y 積分で割らず、白色に合わせた色の正規化も行わない。
測光単位への変換や表示露出はフィルム側の明示的な変換とする。

CIE と太陽の表は線形補間する。レンダラの有効波長域は、この実装では
**360–830 nm** に固定している。初期化時に範囲を検査するので、データの無い領域を
黒体などで自動補完する処理は存在しない。入力波長の数値を各表の公表 nm 座標で
評価し、準備処理に空気中波長と真空波長の変換は含めていない。

### 太陽照度から有限円盤の放射輝度へ

`SolarSample::irradiance_w_m2_nm` は放射照度であり、放射輝度ではない。
角半径 `alpha` の一様な円盤光源なら、太陽中心に垂直な面に対し

\[
  E_\lambda=L_\lambda\int_{\mathrm{disk}}\cos\theta\,d\omega
  =L_\lambda\pi\sin^2\alpha
\]

なので、シェーダは

\[
  L_\lambda=\frac{E_\lambda}{\pi\sin^2\alpha}
\]

を使う。円盤の立体角 `Omega = 2*pi*(1-cos(alpha))` と投影立体角は異なる。
一様円盤方向のサンプリング PDF は `1/Omega sr^-1` であり、上式に重ねて
照度を正規化しない。NEE の推定器ではこの方向 PDF を通常どおり使用する。
この変換はスペクトル loader の処理ではなく、光源の実装側の責務。

### D65 と既存 RGB 面光源

D65 は太陽の代用品には使わない。用途は既存 RGB emission の分光的な意味を
一定にすること。元は [CIE standard illuminant D65](https://www.cie.co.at/datatable/cie-standard-illuminant-d65)
の実 CSV、DOI `10.25039/CIE.DS.hjfjmt59`、**CC BY-SA 4.0** である。

固定区間 `[360,830] nm`、1 nm grid、台形積分で

\[
  D_{65}^{\mathrm{norm}}(\lambda)=kD_{65}^{\mathrm{raw}}(\lambda),\qquad
  k=\frac{\int\overline{y}(\lambda)d\lambda}
  {\int D_{65}^{\mathrm{raw}}(\lambda)\overline{y}(\lambda)d\lambda}
\]

とする。生成時の double 値は次のとおり。

| 項目 | 値 |
|---|---:|
| `trapz(ybar)` | `106.8569149167669` |
| `trapz(D65_raw*ybar)` | `10567.081562022116` |
| `k` | `0.010112244737544995` |

この正規化は「Y 重み付き平均を 1」にするもので、D65 の Y 積分自体を 1 に
する規約ではない。積分範囲を実行時の波長設定に合わせて動的に変更してはならない。

非負の linear RGB 発光値 `e` に対し、M1 で採用する変換は

\[
  s=2\max(e_r,e_g,e_b),\quad
  e'=e/s,\quad
  L_{e,\lambda}=s\,\rho_{\mathrm{RGB2Spec}}(e',\lambda)
                 D_{65}^{\mathrm{norm}}(\lambda).
\]

`s=0` なら発光は 0。RGB 値のスケールはこの emission authoring 規約に属し、
TSIS の測定された放射照度とは独立である。これは RGB ライトに明示的な SPD を
与える方法であり、実在する照明器具の測定スペクトルを復元するものではない。
一般の RGB から物理的に一意なスペクトルは決まらない。

## 3. 公式 RGB2Spec の固定と GPU API

[公式リポジトリ](https://github.com/mitsuba-renderer/rgb2spec) を commit
`721145dedf2491851bd46ab8fd165955cb38ddaf` に固定し、掲載したソースを無改変で
`third_party/rgb2spec/` に保存した。`UPSTREAM.json` に全ファイルの URL と SHA-256 がある。
ライセンスは Wenzel Jakob の **BSD-3-Clause**。`rgb2spec_device.hpp` は
公式 `rgb2spec_fetch()` の移植で、元の著作権表示と変更内容をヘッダに記載している。

これは Jakob & Hanika の研究に基づく公式 self-contained 実装であり、README にある
とおり元論文の supplemental implementation と solver は同一ではない。
M1 は通常の `rgb2spec_fetch()` を使う。追加の最適化を行う `rgb2spec_fetch_opt()` は
呼ばない。異なるアルゴリズムを同じ API 名で隠していない。

反射率は、波長 nm に対する 3 係数で

\[
  p(\lambda)=(c_0\lambda+c_1)\lambda+c_2,\qquad
  \rho(\lambda)=\tfrac12+\frac{p(\lambda)}{2\sqrt{1+p(\lambda)^2}}
\]

と評価する。GPU helper は `fmaf` と `hypotf` を使う。符号や係数順を変更していない。
neutral RGB は波長によらない定数として扱うので、黒・白・中性灰色が厳密に保たれる。

### ホスト

```cpp
#include "m1/spectral_assets.hpp"

auto assets = m1::load_spectral_assets(asset_directory);
m1::RgbToSpectrum upsampler(asset_directory / "srgb.coeff");

const uint32_t resolution = upsampler.resolution(); // 64
const float* scale = upsampler.scale_data();       // borrowed host memory
const float* data = upsampler.coefficients_data(); // borrowed host memory
const size_t nscale = upsampler.scale_count();     // 64 floats
const size_t ndata = upsampler.coefficients_count(); // 2,359,296 floats
```

ホスト `reflectance_coefficients({r,g,b})` は公式関数を呼び、
`reflectance({r,g,b})` は exact neutral を表せる `SpectralColor` を返す。
これらに渡す RGB は **linear sRGB の反射率 [0,1]**。NaN、inf、範囲外は例外とする。
公式実装にある入力 clamp に誤入力を吸収させない。

GPU upload に必要なのは `scale`、`data`、`resolution` のみ。
一度 upload した後、ホストの `RgbToSpectrum` を保持し続ける必要はないが、
borrowed pointer を使う upload が完了するまでは生存させる。
前方モデルの補助 block は GPU の通常 lookup に不要。

### GPU とテクスチャ

```cpp
#include "m1/rgb2spec_device.hpp"

m1::Rgb2SpecTableView table{gpu_scale, gpu_data, resolution};
float rho;
bool ok = m1::rgb2spec_reflectance_device(
    table, linear_rgb.x, linear_rgb.y, linear_rgb.z, wavelength_nm, rho);
```

既存 CUDA RGB texture を維持し、**sRGB デコード後の linear RGB をフィルタし、その
RGB に対して lookup を 1 回行う**。係数をテクセル間でフィルタする方式ではない。
非線形の spectral upsampling とフィルタは可換ではないので、この順序を固定する。
`ok=false` は入力または table 契約の不成立であり、0 の反射率として握りつぶさない。

`rgb2spec_fetch_device(table,r,g,b,out_coefficients)` も提供する。
有効入力での最大成分の tie rule、axis の回転順、trilinear index の順序は
公式 C 実装に合わせている。正の subnormal RGB で `63/z` が overflow する場合だけ
等価な比の計算順に変更し、有限値を保つ。

### テクスチャの sRGB 復号は CUDA で一度だけ行う

`src/texture.cpp` は色テクスチャの符号化済み RGB と、normal / ARM の数値を
CPU 上でガンマ変換せずに読み込む。WIC は `WIC_FLAGS_IGNORE_SRGB` で読み、
DDS などに残る `_SRGB` format は、展開・変換の前に
`ScratchImage::OverrideFormat(MakeLinear(format))` で同じ画素配置の非 sRGB format
へ付け替える。この操作は画素値の線形化ではない。出力 format に `_SRGB` を指定すると
拒否する。既存の呼び出しはすべて `R8G8B8A8_UNORM` を指定している。

用途は既存の `isDiffuseTexture` flag が決める。diffuse と emission は true として
CUDA texture の `sRGB=1` で RGB を一度だけ復号し、normal / ARM は `sRGB=0` で
数値を保持する。alpha は色のガンマ変換を受けない。ファイルのタグの有無を用途の
判定に使わず、色テクスチャは sRGB 符号化、データテクスチャは数値データとして扱う。

この修正はレンダラが固定する DirectXTex commit
`169f7cf3504c7f2c0cca1dbb3ecc515e66add7ef` の公式実装に合わせた。
[`WIC_FLAGS_IGNORE_SRGB` で metadata の sRGB 判定を省く実装](https://github.com/microsoft/DirectXTex/blob/169f7cf3504c7f2c0cca1dbb3ecc515e66add7ef/DirectXTex/DirectXTexWIC.cpp#L569)、
[`MakeLinear` の format 対応](https://github.com/microsoft/DirectXTex/blob/169f7cf3504c7f2c0cca1dbb3ecc515e66add7ef/DirectXTex/DirectXTexUtil.cpp#L1445)、
[`OverrideFormat` が metadata だけを変える実装](https://github.com/microsoft/DirectXTex/blob/169f7cf3504c7f2c0cca1dbb3ecc515e66add7ef/DirectXTex/DirectXTexImage.cpp#L730)、
[`Convert` が `_SRGB` 入力を自動的に復号する実装](https://github.com/microsoft/DirectXTex/blob/169f7cf3504c7f2c0cca1dbb3ecc515e66add7ef/DirectXTex/DirectXTexConvert.cpp#L3119)
を確認した。従来の経路では、タグ付き画像を CPU で線形化してから CUDA でも
復号することがあり、暗くなる原因になっていた。

Windows / DirectXTex / CUDA の実環境での読み込み確認は未実施。受入確認では、
タグあり・なしの同じ 8-bit 中間灰色を読み、CPU の RGB がともに 128 を保持し、
色 texture からの値が約 0.21586、データ texture からの値が約 0.50196 になることを
調べる。DDS の `_SRGB` と対応する `_UNORM` でも同じ規約を確認する。

### モデル形式の検査

M1 の係数は little-endian IEEE binary32 の SPEC ファイルで、解像度は 64。
この固定 commit の forward block は `nfine=288`。ホストはロード前に
ファイル長、magic、解像度、全 float の有限性、scale の単調性と端点を確認する。
同じ magic を持つ別 gamut を誤って使わないよう、forward block の sRGB→XYZ 行列も検査する。
旧形式や別 gamut、異なる解像度はこの API では拒否する。

生の SPEC 形式そのものにソース commit や gamut 名を記録する欄はない。
そのため、正確なソース・係数の SHA-256 と生成条件は manifest で管理し、
prepare script の `--verify-only` で検証する。実行時の C++ loader は構造と数値の検査を
行い、ファイルの暗号学的な同一性の検証は準備段階が担当する。

## 4. ビルドと再現

配布する係数テーブルは生成済みなので、実行時に最適化を走らせる必要はない。
準備・チェックは Python 3.9 以降の標準ライブラリだけで実行できる。

```sh
# この移植単位の配置
python tools/prepare_spectral_assets.py --verify-only

# NeuralPFRenderer の spectrum/m1 配置
python tools/prepare_spectral_assets.py --asset-dir spectrum/m1 --verify-only

# 欠けた公式元データを取得し、canonical CSV を元データから再生成
python tools/prepare_spectral_assets.py --asset-dir spectrum/m1

# 明示的に解像度64の係数を再生成する場合（CMakeとC++ compilerが必要）
python tools/prepare_spectral_assets.py --asset-dir spectrum/m1 --rebuild-model
```

スクリプトは取得済み元データを優先して検証し、存在しない元データだけを公式 URL
から取得する。取得物の SHA-256 が既知の値と違うと停止する。ダウンロード失敗時に
別の太陽、低解像度係数、擬似データへ変更する経路は無い。`--verify-only` はネット接続、
コンパイル、ファイル変更を行わない。版・許諾の snapshot と NOTICE は本パッケージに
含まれる補助ファイルとして保持する。

係数を再生成すると、compiler/architecture による浮動小数点の差でハッシュが変わる
可能性がある。生成した実バイナリのハッシュと compiler の設定を manifest に記録する。
形式や解像度を黙って変えることはない。

CMake 側は `rgb2spec.c` を C 言語でコンパイルし、ホスト adapter にリンクする。
既存プロジェクトが CXX のみを有効にしている場合は C も有効にする。

```cmake
add_library(m1_rgb2spec STATIC third_party/rgb2spec/rgb2spec.c)
target_include_directories(m1_rgb2spec PUBLIC third_party/rgb2spec)
add_library(m1_spectral_assets STATIC src/m1/spectral_assets.cpp)
target_compile_features(m1_spectral_assets PUBLIC cxx_std_17)
target_include_directories(m1_spectral_assets PUBLIC include)
target_link_libraries(m1_spectral_assets PUBLIC m1_rgb2spec)
```

GPU code は `rgb2spec_device.hpp` と小型の `spectral_types.hpp` を使うため、
ホストの filesystem/unique_ptr や upstream の x86 SIMD header を CUDA code に持ち込まない。

### 同梱した実ファイルのハッシュ

| canonical file | SHA-256 |
|---|---|
| `cie_1931_2deg.csv` | `b841c7e91e12167e61eace9468afd82eac3937da98aaecc5b4c0a119d2a6f2a9` |
| `solar_am0.csv` | `1c083f893fdd985c3423f25c73d17ec63aa87b77d46411c9af8e3429f607bb22` |
| `d65_normalized.csv` | `706e21fd9f3a2c542c2071ec77359beac1c636aa7de45d1aa4a020707746f101` |
| `srgb.coeff` | `1bb2e8bc757bdd90f2a8ae74e08011906697fcfa850e6e5534861577a9befeff` |

原本のハッシュ、ソースファイル全件のハッシュ、許諾 notice のハッシュは
`manifest.json` と `third_party/rgb2spec/UPSTREAM.json` に格納してある。

## 5. 実施した検証

`tests/test_m1_spectral_assets.cpp` を公式 C runtime と一緒に g++ 13.3 で
コンパイルして実行した。チェックした内容は以下。

- CIE の既知の行と metadata の Y 列の総和、波長範囲と標本数。
- 太陽の原本の値、float に変換した波長の順序、可視域積分と物理単位。
- D65 の Y 重み付き平均が 1 であること。
- sRGB cube の端点・同値成分・中性色と乱数を含む **10,126 色**について、
  host wrapper と公式関数、および CPU 実行した GPU 共通 lookup の一致。
- 6 波長での評価。公式 evaluator との差の最大値は **7.57e-6**。
  FMA の有無など丸めの差を許容し、同じモデルを評価していることを確認した。
- 実 CIE/D65 を用いる独立な XYZ 積分による 4 色の RGB 再現。
- exact black/white/neutral、入力範囲外・NaN・inf、欠落・破損・別 gamut モデルの拒否。
- `prepare_spectral_assets.py --verify-only` による原本・canonical・係数・根拠・notice の検証。

この検証はホスト実行の数値・形式検証である。CUDA compilation、OptiX launch、
既存レンダラ内の resource lifetime、実画像、GPU 性能は統合作業側で検証する。
ここでそれらまで実行したという意味ではない。

# M3: 球状 Rayleigh 大気のスペクトラル・パストレーシング

対象リポジトリ: https://github.com/Somatic-Cell/nf-renderer

適用基準 commit: `6e4794fb50fe3d9eafaacd78e5fe0b6e2098a103` (`Fix: error handle`)

## 1. 今回の範囲

既存の OptiX レンダラに、球状の指数減衰 Rayleigh 大気を追加します。既存の
ウィンドウ、CUDA バッファ、SBT、材質、RGB2Spec、スペクトル資産、出力を使います。
1 パスにつき 1 波長を保持し、波長 PDF の補償は film への変換時に一度だけ行います。

| 項目 | M3 の実装 |
|---|---|
| 街のポリゴン | M2 のスペクトラル BSDF、RGB テクスチャ、alpha、法線マップ |
| 大気 | 球殻内の指数密度、純散乱の scalar Rayleigh |
| 散乱回数 | パスを継続して多重散乱を追跡。表面と媒質で共通の bounce 上限と RR |
| ポリゴンからの NEE | 有限円盤の太陽、および面光源 |
| 媒質頂点からの NEE | 太陽のみ |
| 地球 | 解析的な球交差、太陽の遮蔽、波長に依存しない Lambert 反射 |
| 光学的厚さ | 誤差推定付きの数値積分をその場で実行 |
| 自由飛行 | 光学的厚さの逆 CDF を数値的に解く |

大気の LUT、ratio tracking、delta tracking、`cuda_driver_wrapper` は追加していません。
既存の旧大気/VDB/ニューラル分岐を有効にする設定ではありません。
rainbow/NF の重み、Mie 散乱、雨粒、虹はこの段階の対象に含みません。

この M3 は、後の高速化とニューラル位相関数の比較に使う**数値参照の実装**です。
Windows アプリ全体のビルド、OptiX の実行、画質と FPS はユーザー環境での確認が必要です。
こちらで実施した検証の詳細は `validation/m3/` に記録しています。

## 2. 適用と起動

ZIP 内のファイルを、リポジトリの同じ相対パスへ配置してください。
`M3_MANIFEST.json` に変更・追加ファイルと SHA-256、`m3-rayleigh.patch` に変更差分があります。
前回までの検査に必要な小さな M1/M2 シーンと M1 のテスト用形状も同梱しています。
Emerald 本体や既存 SDK、学習済み重みは含みません。

差分だけを確認する場合は `git apply --check m3-rayleigh.patch` を使えます。
ZIP を上書き配置した後に同じ差分を適用する必要はありません。
差分は M3 の変更と追加を含み、以前の配布物から再同梱した M1/M2 の検査用資産は ZIP 側にあります。
シーンやモデルは既存の Git ignore 対象なので、今回のプリセットをリポジトリへ登録する際は
`git add -f scene/m3_*.json` が必要です。検査用資産も追跡する場合は同様に対象を明示してください。

実行ファイルとシェーダーは同じソースから再ビルドしてください。共有 launch parameter と
PRD が変更されているため、古い `.optixir` と新しい実行ファイルを混在させないでください。
CMake のシェーダー依存関係に M2 修正と M3 の共有ヘッダーを追加しています。

リポジトリ直下の PowerShell で、これまで使っていた build ディレクトリを指定します。

```powershell
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release
.\build\bin\Release\NeuralPFRenderer.exe m3_rayleigh_sky.json
```

最初の `m3_rayleigh_sky.json` は外部メッシュが不要です。M1 で用意した
`spectrum/m1` のスペクトル資産は引き続き必要です。球面地表と空だけを表示します。
空の初期設定は 512 × 288、1 sample/launch、蓄積ありです。

| シーンファイル | 用途 |
|---|---|
| `m3_rayleigh_sky.json` | 空と球面地表だけの確認。512 × 288 |
| `m3_emerald_day.json` | M2 の地上カメラに Rayleigh 大気を追加。640 × 360 |
| `m3_emerald_overview.json` | 上空から街を確認。640 × 360 |
| `m3_emerald_sunset.json` | 太陽天頂角 88 度の低い太陽。640 × 360 |
| `m3_emerald_vacuum.json` | day と同じ地表・カメラで `beta550=0`。大気だけをなくす対照 |

Emerald の既存配置はそのままです。

```text
model/mesh/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx
```

FBX が参照する DDS も従来と同じ場所に置きます。

```powershell
.\build\bin\Release\NeuralPFRenderer.exe m3_emerald_day.json
.\build\bin\Release\NeuralPFRenderer.exe m3_emerald_overview.json
.\build\bin\Release\NeuralPFRenderer.exe m3_emerald_sunset.json
```

それぞれ別の起動例です。比較時は一つを終了してから次を起動します。
出力解像度の指定は `outputs.resolution` です。未指定の既存シーンは
1280 × 720 を維持します。OS の表示倍率がある場合は、実際の framebuffer ピクセル数を
使って GPU バッファを確保し、そのサイズを保存メタデータに記録します。

## 3. 物理モデルと座標

シーン座標はメートル、上方向は +Y です。地球中心は

\[
C=(0,\;\mathrm{groundY}-R,\;0)
\]

とします。`groundY` は X=Z=0 における球面地表の Y 座標です。
Emerald の M2 監査で得た最低 Y は約 -4.928 m なので、プリセットでは -6 m に設定しています。
形状の位置やスケールを大気用に変更していません。

高さと散乱係数は

\[
h(x)=\lVert x-C\rVert-R,
\qquad
\sigma_s(x,\lambda)=\beta_{550}
\left(\frac{550}{\lambda}\right)^4
\exp\left(-\frac{h(x)}{H}\right)
\]

です。波長は nm、散乱係数は 1/m です。0 ≤ h ≤ `atmosphereHeight` でこの密度を使い、
外側は真空、地表の内側は不透明な地球とします。吸収を含まないので σt=σs です。

| JSON の `rayleigh` 内の項目 | 既定値 | 意味 |
|---|---:|---|
| `enabled` | false | M3 の球面地表・大気を有効にする |
| `planetRadius` | 6371000 | 地球半径 R [m] |
| `atmosphereHeight` | 100000 | 密度を打ち切る地表からの高さ [m] |
| `scaleHeight` | 8000 | 密度のスケールハイト H [m] |
| `groundY` | -6 | 地表の基準 Y [m] |
| `beta550` | 0.000013 | 地表・550 nm の散乱係数 [1/m] |
| `groundAlbedo` | 0.1 | 球面地表の波長に依存しない反射率 |
| `areaNeeTransmittance` | true | 面光源 NEE の接続にも大気の減衰を含める |

これらは明示的な初期モデルの値です。屈折、偏光、オゾン吸収、エアロゾル、雨粒は含みません。
地表の反射率も、都市の外側を含む地球全体について一つの中性値を使います。

Rayleigh の位相関数は、立体角あたりの密度として

\[
p_R(\mu)=\frac{3}{16\pi}(1+\mu^2)
\]

を使います。余弦 μ の周辺 CDF とその逆は

\[
F(\mu)=\frac12+\frac{3\mu+\mu^3}{8},
\qquad
\mu=2\sinh\left(\frac{\operatorname{asinh}(4u-2)}{3}\right)
\]

です。方位角は [0,2π) から一様に選びます。ここは解析的な評価・サンプリングです。
`include/m3/transport.hpp` は位相関数の値とサンプリング PDF を別フィールドで返します。
今回の Rayleigh では両者が一致しますが、将来のニューラル proposal でも混同しないためです。
将来の NF の向き・条件・立体角測度との契約は、この実装だけで確定させていません。

## 4. 光学的厚さと自由飛行

接続区間の光学的厚さと透過率は

\[
\tau(a,b,\lambda)=\int_a^b\sigma_t(x+s\omega,\lambda)\,ds,
\qquad T=\exp(-\tau)
\]

です。レイと外球・地表の交差で積分区間を切り、地球に先に当たる場合はそこで止めます。
大気の入口・出口を scattering bounce として数えることはありません。

CPU と CUDA で共通の `include/m3/atmosphere.hpp` が以下を担当します。

- 地球中心とレイの幾何を double で計算し、地表近傍の高さと短い交差距離の桁落ちを避ける。
- レイの最近接点で区間を分け、それぞれを高さが単調に増える方向の積分にする。
- Gauss 7 / Kronrod 15 の埋め込み則で誤差を推定し、必要な区間を適応分割する。
- 長い区間で狭い高密度領域を見落とさないよう、高さの変化に沿って区間を分ける。
- ごく薄い尾部を省ける場合は、単調性から求めた上界を誤差予算に加える。
- 自由飛行では τ*=-log(1-u) に対し、同じ密度の積分を使って bracket 付き Newton/二分探索を行う。

これは誤差推定付きの数値積分であり、厳密な閉形式積分でも、区間演算による誤差保証でもありません。
積分と逆関数の数値誤差は残ります。未収束、非有限値、予算超過を真空や黒へ置換しません。

OptiX の float 方向は、normalize 後でも double 上で厳密な単位長とは限りません。
共通 API は元の方向と `t_max` を受け取り、方向の長さを使って物理距離 [m] に換算します。
返り値の `distance_m` と OptiX のレイ・パラメータ `ray_t` は区別しています。

地平線付近では、球の半径とレイの最接近距離を直接引くと小さな高度が失われます。
地表からの局所座標を使った二次方程式と補償付き判別式で交差を求め、密度積分の最接近高度も
同じ判別式から復元します。独立の long double 計算と比較した接線近傍 126 ケースを回帰試験に含めています。

また、地表のすぐ外にある double の散乱点が、OptiX 用の float へ丸まるだけで地表内へ入ることがあります。
`include/m3/position.hpp` は元の double 点が地表外であることを検証したうえで、通常の float 変換結果が
内側になった場合だけ、内向きに丸まった成分を外向きの隣接 float 値へ移します。
変更は各座標につき最大 1 ULP で、正常な変換結果は変更しません。真に地表内の点は失敗として扱います。
これは float 座標の表現上の補正です。媒質散乱点へ固定距離のオフセットを毎回加える処理ではありません。

### 透過率を二重に掛けないための規則

自由飛行は、区間を通過する離散確率 T と、途中で散乱する連続密度 σs T を含む分布です。
純散乱の場合、無衝突で表面・太陽へ着いた枝にも、途中で散乱した枝にも、距離サンプリングによる
追加の throughput 係数は 1 です。到達した枝で T をもう一度掛けると、平均が T² になってしまいます。

明示的にサンプルした太陽・面光源への NEE は自由飛行をサンプルしないため、こちらの接続には T を掛けます。
NEE と BSDF/phase continuation の MIS には、同じ頂点に到達した条件での方向 PDF を使います。
RR の生存確率を片側だけの PDF に入れることもしません。

### 数値パラメータ

| 項目 | 既定値 | 対象 |
|---|---:|---|
| `tauAbsoluteTolerance` | 1e-10 | 光学的厚さの絶対誤差目標 |
| `tauRelativeTolerance` | 1e-8 | 光学的厚さの相対誤差目標 |
| `inverseTauTolerance` | 1e-9 | 自由飛行の τ 残差と積分誤差の和 |
| `distanceAbsoluteTolerance` | 0.001 | 自由飛行距離の絶対誤差目標 [m] |
| `distanceRelativeTolerance` | 1e-10 | 積分区間長に対する距離の相対誤差目標 |
| `maxQuadratureDepth` | 16 | 適応分割の深さ上限 |
| `maxQuadratureEvaluations` | 4096 | 一回の光学的厚さ評価での密度評価数の上限 |
| `maxInverseIterations` | 64 | 逆 CDF の反復上限 |

1 mm は数値ソルバの目標です。元の三角形、OptiX の交差点や継続レイの位置は float なので、
大きなシーン座標で最終的な位置の精度が常に 1 mm になるという意味ではありません。
非常に薄い高高度で距離の目標を厳しくしすぎると、τ の丸め限界によって達成できなくなります。
内部積分が達成不可能な丸め以下の精度を要求しないよう調整しつつ、返却前にはユーザーが指定した
τ 残差と距離の条件を改めて検査します。

### Yue (2024) をここで使わない理由

以前のご質問に対応する論文は Yue (2024), *Analytical approximation of the definite Chapman
integral for arbitrary zenith angles* です。この論文は有限区間・任意天頂角の Chapman 積分の
閉形式近似を扱います。元の指数密度に対する厳密な原始関数ではありません。
また、自由飛行距離の逆関数まで閉形式で与える手法ではありません。[1]

M3 では `solver: "numerical_reference"` だけを受理します。高速化で Yue 近似を追加する場合は、
今回の数値実装と比べて、短区間、地平線、高高度、自由飛行 PDF との整合性を検証する基準ができます。
近似実装を入れていないのに設定名だけを受理する分岐はありません。

## 5. 面光源の扱い

表面から太陽への NEE は、ポリゴンと地球による遮蔽、および大気の T を含みます。
媒質の散乱点も太陽について同じ処理を行います。太陽円盤内のサンプル方向ごとに地球遮蔽を評価します。

表面から面光源への NEE は、既定の `areaNeeTransmittance: true` で T を含みます。
M3 の Rayleigh 減衰計算はネットワークを呼び出しません。

ご提案の性能上の選択を試す場合は、`areaNeeTransmittance: false` にできます。
この場合、面光源 NEE の媒質減衰だけを省き、ポリゴンと地球の遮蔽は残します。
**継続パスは通常の媒質を通るままなので、両戦略の被積分関数が異なり、結果は MIS 設定にも依存する近似になります。**
この選択を物理的に整合した推定器としては扱いません。保存 JSON にこの設定も記録します。
M3 の基準画像と MIS 比較では true を使用してください。

媒質の散乱点からは面光源 NEE を行いません。一方、位相関数をサンプルした継続パスが面光源へ
到達することは許可します。そのときは競合する NEE 戦略がないため、発光寄与の MIS 重みは 1 です。
これは `m1DirectMode: "nee"` の比較設定でも同じです。

三角形は各パス区間で従来どおり一回 trace し、受理された最も近い表面までの距離を使って大気を処理します。
そのため、alpha any-hit で選ばれた表面を再 trace して乱数で選び直すことはありません。
表面からの shadow ray は従来の原点オフセットを使い、T も実際の shadow ray と同じ区間で評価します。

## 6. 確認手順と保存

まず空だけのシーンを蓄積し、空、大気の地平線、地表と太陽の関係を確認してください。
次に day と vacuum を同じ露出・カメラで比べます。day では空からの照明が加わり、遠い物体には
大気の減衰と散乱光が加わります。これらはモデルから期待される変化で、ここで GPU 画像を実測した結果ではありません。

`m1DirectMode` の `mis` / `bsdf` / `nee` は既存の名前を維持しています。
`bsdf` は媒質内では phase continuation のみ、という意味も持ちます。
太陽円盤は非常に小さいため、continuation だけの比較は多くのサンプルを必要とします。
数フレームの PNG の見た目だけで三モードの一致を判断しないでください。

多重散乱の上限は `maxBounce: 16` です。厳密な無限回散乱の解ではなく、有限 bounce で切ったパスの推定です。
高密度や低い太陽で基準画像を作る際は、同じ設定で bounce 上限と spp を増やし、収束を確認してください。
RR は表面と媒質で共通です。数値精度の設定を変更した比較では、露出や材質も同時に変更しないでください。

既存の保存操作から PNG と `.linear.pfm`、`.linear.json` が得られます。
比較は tonemap 後の PNG に加え、線形 PFM を使ってください。
JSON には大気の全パラメータ、初期モデル、太陽、波長範囲、spp、bounce、減衰省略の有無を記録します。

異常値や未収束を検出するとフレーム全体を拒否し、`m1_failure.log` に最初の診断を残します。
ファイル名は既存のままですが、M3 のステージ、理由、数値ソルバの status、乱数、レイ、波長、
反復数、誤差、大気設定が記録されます。件数だけを無視して受理する変更はしていません。

## 7. CPU 検証を再実行する

GPU、OptiX、OpenGL を使わず、共通の数値コードを検査できます。
既存 `ext/json` がある場合の例です。

```powershell
cmake -S tests/m1 -B build-m3-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-m3-cpu --config Release
ctest --test-dir build-m3-cpu -C Release --output-on-failure
```

JSON ヘッダーが別の場所なら `-DM1_JSON_INCLUDE_DIR=...` を指定します。

`test_m3_atmosphere.cpp` は球殻・地球遮蔽、垂直区間の解析解、独立の数値参照との比較、
短区間、地平線、方向の距離換算、高高度の逆 CDF、意図的に達成不能な精度の失敗を検査します。
`test_m3_transport.cpp` は位相関数の正規化とモーメント、CDF、サンプルと PDF の一致、
無衝突確率、二重減衰の回帰、媒質から面光源へ到達した場合の重み、太陽 MIS の期待値を検査します。
`test_m3_position.cpp` は実際の自由飛行で得た地表近傍の点、球面上の異なる向き、1 ULP の変換、
不正点の拒否を検査します。
従来の BSDF、area PDF、診断、alpha、接線、スペクトル、シーン読み込みの検査も残しています。

## 8. 主要ファイル

| ファイル | 役割 |
|---|---|
| `include/m3/parameters.hpp` | ホスト・デバイス共通の物理・数値設定 |
| `include/m3/atmosphere.hpp` | 球殻交差、密度、光学的厚さ、T、自由飛行 |
| `include/m3/transport.hpp` | Rayleigh 評価・解析サンプリング・MIS 戦略の区別 |
| `include/m3/position.hpp` | 媒質散乱点を地表外の float 座標として表現する |
| `shaders/device/m3_atmosphere.cuh` | OptiX から共通大気コードへの接続と診断 |
| `shaders/device/m1_surface.cuh` | 表面・媒質・地表を統合したパス、太陽・面光源 NEE |
| `include/sceneDescIO.hpp` | 大気設定の読み書き、値と組み合わせの検証 |
| `include/launch_params.h` / `src/renderer.cpp` | パラメータの転送、空シーン、診断 |
| `include/application.hpp` / `src/application.cpp` | 表示、初期解像度、保存メタデータ |
| `viewer/glfw_window.hpp` / `.cpp` | 指定した初期サイズでのウィンドウ作成 |

## 参考文献

1. D. Yue, 2024. [Analytical approximation of the definite Chapman integral for arbitrary zenith angles](https://acp.copernicus.org/articles/24/5093/2024/). Atmospheric Chemistry and Physics, 24, 5093–5097. DOI: 10.5194/acp-24-5093-2024. 今回は近似式を実装していません。
2. Pharr, Jakob, Humphreys. [Sampling Volume Scattering](https://pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering), PBRT 3rd edition. 媒質サンプリングの連続密度と表面到達の離散確率、throughput の一般的な背景。今回の逆 CDF 数値ソルバは本プロジェクトの実装です。
3. Bruneton. [Atmospheric scattering functions](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html). scalar Rayleigh 位相関数と大気光学の定義の参照。M3 はこの実装の前計算・LUT 手法を使用していません。

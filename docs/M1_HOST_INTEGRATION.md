# M1 ホスト統合と確認範囲

M1 は既存の `Renderer`、CUDA バッファ／テクスチャ、OptiX の GAS／IAS、SBT、
ウィンドウと画像出力の経路に追加した表面用スペクトル積分器である。
**Windows アプリ全体のビルド、GPU 起動、画像の正しさと性能は未検証。**
ここでいう確認済みは、以下に明記したソース照合とヘッダのホストコンパイルを指す。

## 1. 変更した接続

主な対象は [renderer.cpp](../src/renderer.cpp)、[renderer.hpp](../include/renderer.hpp)、
[launch_params.h](../include/launch_params.h)。シェーダとの追加共有データは
[M1RenderParams](../include/m1/render_params.h) にまとめた。

| 接続 | 変更内容 |
|---|---|
| 初期化 | `m1Surface=true` で spectral entry を選ぶ。VDB、環境マップ、旧巨大地面球の指定は明示的に拒否する。M1 では VDB／大気 LUT／Mie 表／ネットワーク重みを読み込まない。 |
| マテリアル | `createTextures()` を `loadAssets()` より前に移し、作成済み texture handle を material table に格納する。 |
| スペクトル | 実 CIE、AM0、正規化 D65、公式 RGB2Spec の scale と係数配列を既存 `CUDABuffer` でアップロードする。資産ディレクトリは既定で `spectrum/m1`、作業ディレクトリ相対。 |
| UV | diffuse／normal／emissive の UV ポインタを分ける。面光源にも emissive UV と alpha 用 diffuse UV を別々に保存する。 |
| alpha | 三角形 GAS を `REQUIRE_SINGLE_ANYHIT_CALL`、表面 instance を `NONE` とし、既存 any-hit を有効にする。 |
| フィルム | 既存 float4 color buffer の線形 RGB を `downloadLinearPixels()` で取り出す。表示用トーンマップを通さない。 |
| 診断 | path／film の不正値を数える device counter を各 launch 前にゼロにし、完了後に検査する。0 以外ならフレーム全体を失敗として例外にする。 |

`LaunchParams` はゼロ初期化する。M1 が使わない旧スペクトル texture や大気、
ネットワークのポインタを、不定値のままシェーダへ送らない。
太陽の天頂角／方位角は JSON の度数から既存 `sunParams` のラジアンへ変換し、
GUI で方向を変更したときは蓄積をリセットする。

M1 が実行時に呼ばない旧ネットワーク用ヘッダも C++ の構文解析は受けるため、
`nsf_hyper_coopvec_pack.hpp` の `CUDA_CHECK` 4 箇所をこのリポジトリの 1 引数 macro に
揃えた。続いて見つかった `safetensors_loader.hpp` の `<iostream>` include 欠落も修正した。
これはヘッダのビルド阻害要因の修正であり、ネットワーク形式や推論の変更ではない。

## 2. メッシュ、SBT、光源の ID を照合した結果

`buildAccel()`、`loadAssets()`、`buildSBT()`、`createLightTable()` はいずれも
`models` の順、そのモデル内の `meshes` の順に走査する。

| 値 | 規約 |
|---|---|
| global mesh index | 全モデルを通した連番 `flatMesh`。非発光メッシュも連番に含める。 |
| IAS instance ID | `flatMesh`。既存実装どおり、メッシュごとに GAS と instance を作る。 |
| SBT record | `flatMesh * 2 + rayType`。radiance と shadow の 2 種類。 |
| material index | それ以前のモデルの material 数の合計＋当該 mesh の local material ID。 |
| texture handle | それ以前のモデルの texture 数の合計＋当該 material の local texture ID。面光源にも同じ offset を適用する。 |
| 発光三角形の識別 | `meshIndex, instanceID, primitiveIndex` の三つ組。BSDF で発光面に到達したときに同じ光源と PDF を再現する。 |

M1 では空の mesh、範囲外 index、属性配列の長さ不一致、非有限／特異な instance
変換を GPU 確保前に拒否する。roughness／metallic texture は現段階では
diffuse と同じ UV set が必要で、異なる場合は拒否する。
元 Assimp loader が存在しない UV set をゼロ UV に置き換える処理は残っているため、
この検査だけで未知の FBX の UV／材質対応を監査したことにはならない。

## 3. 変換と面光源の PDF

光源の頂点には**発光モデル自身の** `modelMatrix` を適用する。
面積と幾何法線は変換後の頂点から求める。受光面の instance 変換を再利用しない。
旧 RGB／spectral の triangle-light callable も、この world-space 頂点を再変換せずに使う。

M1 の emitter-hit シェーダも OptiX の instance 変換で world-space 頂点を取得し、
同じ頂点の外積から面積／表裏を評価する。非一様 scale、回転、平行移動について、
NEE と emitter hit の幾何量の定義は一致する。

負の scale は許容する。**表側は変換後の頂点 winding で定義するため、反射変換で
winding が反転すると、片面発光の側も反転する。** 自動の winding 補正は行わない。
補間法線は inverse-transpose で変換し、幾何法線と同じ側に合わせる。
接線は vector 変換後に再直交化し、bitangent の向きも変換後の向きと照合する。
この処理は M1 シェーダのソース照合による確認で、負の scale の GPU 画像試験は未実施。

M1 の light table は三角形のみ。三角形数を `N` として選択確率は `1/N`、
方向密度は `distance² / (N * worldArea * cosLight)` であり、NEE と emitter hit で共通。
太陽は別の方向サンプルを 1 個生成するため、この `1/N` を太陽 PDF に掛けない。
発光面自身の alpha は NEE の発光値に掛け、BSDF-hit の stochastic any-hit と同じ
平均の coverage にする。

`m1_area_mis.json` は 2 個の `lamp.obj` を異なる変換で配置する。
元の 1 m² の四角形に対し、設定上の面積は A が 1.12 m²、B が 0.77 m²。
これは入力頂点と scale からの期待値であり、Assimp／OptiX 実行結果の測定ではない。
`m1_sun_mis.json` は Courtyard のみで、面光源数は 0 になる構成。
この場合、両 light buffer を空のままにしてポインタを null、count を 0 にする。
シェーダは `count > 0` の場合だけ面光源を読み、空 vector の CUDA upload は行わない。

## 4. バッファと実行順

スペクトル配列は `vector<T>.size() * sizeof(T)`、RGB2Spec は要素数×`sizeof(float)`
で確保し、同じ要素数で upload する。共有 POD のホストサイズは CIE が 16 bytes、
solar と D65 が各 8 bytes。RGB2Spec は resolution 64、scale が 64 floats、
係数が 2,359,296 floats。ローダーの検査済み配列だけを渡す。

GAS／IAS は既存の default stream と同期処理を維持する。レンダー用 stream は
既存の `cudaStreamCreate()` による通常の stream であり、今回 nonblocking stream や
per-thread default stream の設定を追加していない。launch params と診断 counter の
upload 後に `optixLaunch()` を行い、M1 は明示的にその stream の完了を待ってから
counter を download する。線形フィルム／PNG の download も同じ stream を同期する。
この stream 設定を将来変える場合、転送と launch の依存関係を再確認する必要がある。

フレームバッファは積を `size_t` で求め、既存の int pixel index と byte 数に収まることを
検査する。診断は 1 path あたり最大 1 件と最終フィルム検査 1 件を考慮し、
`pixelCount * (spp + 1) <= UINT32_MAX` を要求する。

## 5. OptiX の stack 設定

既存の pipeline と program groups を保ち、固定各 2 KiB の stack 設定を
OptiX 9 SDK の `optixUtilAccumulateStackSizes()` と `optixUtilComputeStackSizes()`
による計算に置き換えた。実行時に算出 bytes をログに出す。

- trace depth: 既存の **4**。M1 の逐次継続と closest-hit 内の shadow trace を含む。
- continuation callable depth: **0**。該当する call site はない。
- direct callable depth: **1**。lens／BSDF／light callable が別の OptiX callable を呼ぶ入れ子はない。
- traversable depth: **2**。単一 IAS → GAS。

## 6. 実施済み／未実施

| 項目 | 結果 |
|---|---|
| global mesh／material／texture offsets、ID、UV、world-space light のソース照合 | 実施。上記の規約で整合。 |
| `LaunchParams`、mesh／light records、M1Params、spectral POD の host compile | g++、C++17、実 CUDA 13.0／OptiX 9 headers で成功。trivially-copyable、standard-layout、alignment と spectral POD サイズを static assertion で確認。 |
| OptiX stack helper の 3 API の host compile | 同じ実 headers で成功。 |
| 旧 CoopVec pack＋checkpoint ヘッダの host compile | 実 CUDA 13.0／OptiX 9／nlohmann JSON headers で成功。macro 引数数と直接 include の修正後、追加の API 型／構文エラーなし。 |
| 差分の whitespace check | 新規エラーなし。 |
| `src/renderer.cpp` 全体の host compile | 試行したが、既存依存の `DirectXTex.h` 不在で停止。成功とは扱わない。 |
| Windows／Visual Studio 2022 での full build、実 CUDA／OptiX 初期化、GAS／SBT 実行 | **未実施**。 |
| 3 手法の画像収束比較、alpha／非一様・負 scale の画像、GPU memory／性能測定 | **未実施**。 |

POD compile では既存 `optix8.hpp` のコメント末尾のバックスラッシュによる
`-Wcomment` だけを warning のまま許容し、それ以外は `-Wall -Wextra -Werror`。
旧ネットワークヘッダの単独 probe では、呼ばない既存 static helper に対する
`-Wunused-function` も抑制した。probe はコンパイルだけで、GPU API を実行していない。
これは full host build の代替ではない。PTX／OptiX-IR、スペクトルおよび BSDF の
別試験結果は、各担当の検証記録を参照する。実 Emerald の読み込み・材質監査は
この M1 fixture の確認範囲に含めない。

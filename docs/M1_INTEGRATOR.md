# M1: NeuralPFRenderer の単一波長 surface integrator

## 実装と検証の範囲

M1 は既存 `NeuralPFRenderer` の OptiX pipeline、SBT、Assimp geometry/material/texture、instance transform、camera callable、GUI framebuffer を使用する。別レンダラを起動する実装ではない。`LaunchParams.m1.enabled` のときだけ、既存と同名の spectral raygen / closest-hit / miss entry から `shaders/device/m1_surface.cuh` に分岐する。RGB entry と非 M1 の旧 spectral code は残している。

M1 の伝搬媒体は真空であり、全区間の transmittance は 1 である。Rayleigh、大気球殻、雨粒、NF、OptiXCoopVec、LUT の評価はこの integrator に接続していない。M1 の画像を、大気または虹のレンダリングの検証結果として扱ってはいけない。

CPU の BSDF 数値検証は 9 群が通過した。実 CUDA 13.0.88 / OptiX 9 headers / 実 NanoVDB headers により、既存 CMake 対象の 14 OptiX translation units を OptiXIR、2 CUDA kernels を PTX にコンパイルできた。**OptiX runtime による pipeline 作成・GPU launch・画像比較は、この環境では実施していない。** ソースをコンパイルできたことと、GPU で正しくレンダリングできたことは別の検証段階である。

結果と正確なコンパイルコマンドは次に保存している。

- `validation/m1_shader_compile.json` / `.log`: 各 translation unit のコマンド、終了値、出力、ソースと生成物の SHA-256、使用ヘッダの最終 SHA-256。
- `validation/m1_bsdf_tests.json` / `.log`: CPU テストのコマンド、出力、使用ソースの SHA-256。

GPU コンパイルは C++20、`compute_86` / `sm_86`、`--cudart none`、`--relocatable-device-code=true` を使用した。fast math、一時的な強制 include、SDK を模した stub は使用していない。旧コード由来の未使用変数等の警告はログに残している。

## 1 path = 1 wavelength

カメラサンプルごとに、設定した波長区間から単一の波長を一様に選ぶ。M1 の入力データ範囲は 360–830 nm で、各公式表の波長座標を使う。空気中／真空の波長変換は加えていない。波長はパスの途中で変えない。Hero Wavelength Spectral Sampling、波長 packet、二次波長の重み付けは行わない。

`PRDSpectral.albedo` を単一波長の scalar throughput `beta`、`contribution` を積算 spectral radiance として使う。波長 PDF による補償は raygen の最後に一度だけ行う。

\[
XYZ = \frac{L(\lambda)}{q_\lambda(\lambda)}
       \bigl(\overline{x}(\lambda),\overline{y}(\lambda),\overline{z}(\lambda)\bigr).
\]

CIE の表は波長について線形補間する。683 を掛ける測光量への変換や、CIE の Y 積分による画像全体の正規化は行わない。旧 spectral raygen の経験的な `0.1` 倍は M1 にはない。XYZ を D65 white point の linear sRGB に変換し、既存 `frame.colorBuffer` に蓄積する。Monte Carlo サンプルの負の linear RGB 成分は正常な場合があるため、積算段階で clamp しない。

RGB reflectance texture は既存 CUDA texture の sRGB decode と線形フィルタを通した **linear RGB** を、Jakob–Hanika RGB2Spec の公開係数表でスペクトルに変換する。この順番は「各 texel のスペクトルを評価してから補間」とは一般に一致しない。M1 は既存 texture path を維持するため、**RGB をフィルタしてから upsample するモデル**を明示的に採用している。HDR 値、非有限値、負の RGB、反射率の入力域外を隠して clamp する処理はない。

## 材質と BSDF

共通 CPU/CUDA 実装は `include/m1/bsdf.hpp` にある。continuous PDF と、理想鏡面イベントの discrete branch probability を別のフィールドで返す。`Null` は正当なゼロ寄与イベント、`Invalid` は入力または数値演算のエラーである。

### Diffuse

反射率を \(\rho(\lambda)\) とする Lambert BRDF と cosine hemisphere sampling を使う。許容入力は \(0\leq\rho\leq1\)。sample weight は \(\rho\) となる。

### 既存の principled / metallic 材質

既存モデルの実体を維持する。

\[
f = (1-m)\frac{\rho}{\pi} + f_{GGX}(F_0,\alpha),
\qquad F_0 = 0.04(1-m)+m\rho.
\]

これは完全な Disney BSDF ではなく、Lambert と単一散乱 GGX の加算モデルである。clearcoat、sheen、rough transmission、複数回 microfacet 散乱、層間の energy compensation はない。加算された diffuse と specular の和が常に energy conserving であるとは主張しない。M1 ではこれを別の材質へ黙って置換していない。

roughness と metallic は RM texture の G/B、または material の定数を使用する。入力域はいずれも [0,1]。旧レンダラの regularization を、名前を付けた `legacy_principled_alpha()` として維持する。

\[
\alpha=\max(r^2,10^{-3}).
\]

これは材質の定義であり、roughness=0 でも delta mirror にはならない。単独の GGX API は直接指定した \(\alpha\in[10^{-4},1]\) を受け付け、範囲外をエラーにする。

GGX は isotropic Trowbridge–Reitz NDF、height-correlated Smith masking-shadowing、Schlick Fresnel を使用する。Schlick は F0 から角度依存性を近似するモデルであり、複素屈折率の実測値から計算した conductor Fresnel ではない。Heitz の visible-normal sampling を用い、反射方向の PDF に正しい reflection Jacobian を含める。[GGX/Smith の定義](https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory)、[Heitz 2018](https://jcgt.org/published/0007/04/01/) を参照。

metallic mixture の sampling probability は diffuse が \((1-m)/(2-m)\)、GGX が \(1/(2-m)\)。どちらをサンプルしても、その方向で **両 lobe の和の f と、両 sampling PDF の mixture** を評価して throughput を更新する。visible normal が macrosurface の下へ反射する場合は Null とし、受理されるまで再サンプルしたり、受理方向の PDF を再正規化したりしない。

### Glass

既存ガラスの分散式

\[
n(\lambda)=1.444829+3506.29/\lambda^2
\]

を維持し、真空とガラスの理想界面として扱う。波長の単位は nm。元の Schlick による branch probability を、非偏光の厳密な dielectric Fresnel と全反射判定に揃えた。反射 branch の sample weight は 1、透過 branch は radiance transport の \((\eta_i/\eta_t)^2\)。transmission の continuous PDF は 0 であり、branch PMF と混同しない。[PBRT の dielectric BSDF](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF) を参照。

内外は幾何法線とレイの向きで判定する。nested dielectric media、ガラス内部の吸収、rough glass、薄板の複数内部反射を一つのイベントへ畳み込むモデルは含めていない。

## Smooth normal と normal map

既存の頂点法線、tangent、normal texture を利用する。法線は inverse transpose、tangent は vector transform を用い、変換後に tangent を法線へ直交化する。normal texture の UV は diffuse UV と独立した `normalTexcoord` を使用する。既存の normal texture 規約に合わせ、signed XY、green の符号反転、positive Z の再構築を行う。必要な normal、tangent、UV が欠ける場合はエラーとし、normal map を無言で無効にしない。

BSDF と cosine は shading normal \(N_s\) の座標系で評価する。一方、反射・透過が幾何的にどちら側へ進むかは \(N_g\) で制限する。この制限を NEE と BSDF sampling の双方へ適用する。支持領域から外れた sample は Null であり、PDF を受理条件付きへ変更しない。

これは camera から辿る radiance transport の規約であり、importance transport に必要な adjoint shading-normal correction を余分に掛けない。通常の shading normal モデル自体が、幾何法線だけの物理的界面とは異なる近似である。将来 BDPT や light tracing を追加する場合は同じ式を無条件に転用せず、adjoint を扱う必要がある。[PBRT の BSDF と幾何法線](https://pbr-book.org/3ed-2018/Materials/BSDFs)、[shading-normal correction](https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation)、[Veach, Non-symmetric Scattering](https://graphics.stanford.edu/papers/non-symmetric/) を参照。

## 光源、NEE、MIS

各 non-delta surface vertex で、太陽を 1 回、面光源集合を 1 回サンプルする。sun と area は独立した sampling strategy であり、太陽を `numLights` の一つとして数えない。M1 の `lightDefinition` は三角形光源だけを含み、選択 PMF は `1/numLights`。

三角形は面積について一様にサンプルし、solid-angle PDF へ変換する。

\[
p_A(\omega)=\frac{1}{N}\frac{d^2}{A\,|N_L\cdot(-\omega)|}.
\]

光源表の vertices、normal、area は **光源自身の instance transform を適用した world space** である。受光面の instance transform を再度掛けない。emitter-hit の逆 PDF は mesh / instance / primitive ID で同じ表の光源を見つけて計算する。emission は幾何法線の表側のみで有効。

太陽は一様 radiance の有限円盤で、方向は既存の sun zenith / azimuth を使う。半径を \(a\)、法線面の分光照度を \(E_\lambda\) とすると、

\[
\Omega_s=4\pi\sin^2(a/2),\qquad
p_s=1/\Omega_s,\qquad
L_{s,\lambda}=\frac{E_\lambda}{\pi\sin^2 a}.
\]

分母の projected solid angle と sampling PDF の solid angle は区別する。表には実データの TOA solar irradiance を用い、D65 を太陽スペクトルの代用にしない。小角度で `1-cos(a)` を直接引かず、half-angle の式と chord distance を用いる。support を広げる恣意的な tolerance や PDF floor はない。

RGB emissive material は測定された SPD ではない。M1 では明示的に、\(s=2\max(R,G,B)\)、\(\rho_c(\lambda)=RGB2Spec(RGB/s,\lambda)\) として、

\[
L_{e,\lambda}=I\,s\,\rho_c(\lambda)D_{65,n}(\lambda)
\]

を使用する。黒は 0。D65 は固定の 360–830 nm 範囲で \(\int D_{65,n}\overline y\,d\lambda/\int\overline y\,d\lambda=1\) に正規化している。強度 \(I\) は既存 `lightIntensityFactor`。RGB texture の値がそのまま物理的な分光放射輝度になるという主張ではなく、D65 を用いる RGB illuminant のモデルである。emissive texture は linear RGB として emissive 定数に乗算し、NEE と emitter-hit で一致させる。

`directMode` は次の意味を持つ。

| 値 | NEE | BSDF からの emitter-hit |
|---|---|---|
| 0: MIS | power heuristic | 相補の power heuristic |
| 1: BSDF only | 実行しない | 重み 1 |
| 2: NEE only | 重み 1 | 直前が continuous event なら抑制。camera / delta 直後は重み 1 |

すべての emitter-hit で `contribution += beta * Le * weight` として加算する。既存の寄与を代入で上書きしたり、throughput を掛け忘れたりしない。delta BSDF には continuous NEE を適用しない。emitter-hit と NEE の MIS の基本構造は [PBRT の path tracer](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer) と同じ測度の扱いに従う。

## Alpha、shadow ray、RR

surface と shadow の双方で同じ diffuse texture alpha を stochastic coverage として使う。`REQUIRE_SINGLE_ANYHIT_CALL` により、同じ primitive が一つの ray に対して複数回 alpha 判定されることを避ける。shadow PRD の RNG は初期化して使用する。

面光源を NEE でサンプルするときは、光源自身の alpha を emission に掛ける。emitter-hit は any-hit が alpha の確率で受理するため、受理後に alpha を二重に掛けない。光源 alpha の UV は diffuse UV、emission の UV は emissive UV を使い分ける。

M1 の shadow ray では glass を無条件透過させない。通常の直線 NEE 接続には屈折を解く処理がないため、glass は遮蔽物となり、その先の経路は BSDF でサンプルされた屈折パスが扱う。M1 の shadow ray は `MASK_SURFACE` であり、大気やネットワークへのクエリはない。

RR は BSDF weight を掛けた後、`rrDepth` 以降で行う。継続率は \(\min(0.95,\beta\,\eta_{scale})\)。透過ごとに、radiance の eta-squared factor を相殺する値を RR 専用の `etaScale` に掛ける。生存した throughput だけを継続率で割る。

NEE と emitter-hit の両方の MIS には、**RR 継続率を含めない同じ条件付き方向 PDF**を使用する。RR の期待値補償は throughput だけで行うので相補性を壊さない。一方の PDF にだけ継続率を掛けてはいけない。

有限の `maxBounce` は scattering vertices の上限であり、有限深度に切った積分である。上限に達した後の endpoint emission は評価する。ray-origin offset は scene-unit epsilon と最低 1 ULP の移動を使う近似で、厳密な floating-point error bound に基づく offset ではない。

## AOV と失敗の報告

既存 GUI の normal / albedo view 用に first-hit AOV を各 sample、各 frame で平均する。miss は両方 0、light の albedo は 0、glass の albedo は 1、通常の surface albedo は linear RGB の base color。normal は first-hit の world-space shading normal である。

非有限な throughput、radiance、PDF、invalid material/texture input、AOV/film accumulation の overflow を検知した場合は診断カウンタを増やす。host は counter が非ゼロなら frame 全体をエラーとして拒否する。サンプルを黒へ置き換えて正常な結果として返す方法や、PDF floor で問題を隠す方法は採用していない。

## CPU 数値結果

| 検証 | 得られた値 | 比較先 |
|---|---:|---:|
| GGX、正面、alpha=1、F0=1 の directional albedo | 0.3068528424 | 解析値 1−ln(2) = 0.3068528194 |
| 同条件の反射方向 acceptance | 0.5 | 解析値 0.5 |
| MIS と RR を併用した Lambert、rho=0.7 | 0.7004681698 | 0.7 |
| 既存 metallic mixture の directional albedo | MC 0.6167723749 | 独立方向積分 0.6164091377 |
| 同 mixture の acceptance | 0.9563331604 | PDF の独立積分 0.9557935299 |
| Ns と Ng を 60 度傾けた cosine sampling の acceptance | 0.7499885559 | 解析値 0.75 |

ほかに Lambert の正規化、GGX の grazing と alpha=1e-4、local BSDF の相反性、eval/sample/PDF の一致、Fresnel/全反射/Snell/eta-squared、入力域外の拒否を確認している。これらは integrator の GPU runtime、scene loading、NEE の画像収束、実 Emerald scene の見え方を検証した結果ではない。

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude \
    tests/test_m1_bsdf.cpp -o test_m1_bsdf
./test_m1_bsdf
```

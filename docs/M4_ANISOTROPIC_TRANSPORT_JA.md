# M4: 雨滴 NF と異方性 RTE の接続規約

この文書は新しい推論モジュールの接続仕様であり、M3 の輸送ループへの統合完了を意味しない。
モデルの物理的な近似精度を教師 CDF に対して再検証する作業は含めない。

## 1. 構成

- 鉛直に配向した同一形状の雨滴を基準とし、その単一粒子の条件付き位相関数を NF で表す。
- 「鉛直」は粒子の配向である。光の入射方向を鉛直の一条件に固定する意味ではない。
- NF の条件は波長と、向きのある粒子軸に対する符号付き入射余弦。出射方向の立体角について正規化される。
- MLP の入力・重み・バイアス・出力は FP16 の CoopVec 行列積。SiLU、RQS、HG、PDF は FP32。
- レンダラの ray/throughput/PDF/MIS と既存の delta/ratio tracking は FP32 を維持する。
- FP16 の重み・活性化の丸め以外に、RQS の層数、bin 数、入力 encoding、activation を変更しない。

## 2. 物理量と方向

本実装の `ki`, `ko` は両方とも物理的な伝播方向である。BRDF の「両方が頂点から外向き」規約とは違う。

`P(ko | ki, lambda)` は単一雨滴の NF PDF [sr^-1]。
`Csca(ki,lambda)`, `Cext(ki,lambda)` は断面積 [m^2]。
`n(x)` は個数密度 [m^-3]。

```
sigma_s(x, ki) = n(x) * Csca(ki)
sigma_t(x, ki) = n(x) * Cext(ki)
K(x; ki, ko)   = n(x) * Csca(ki) * P(ko | ki)
```

共有カタログは `Csca=Cext=Aproj; Cabs=0` という非吸収・幾何断面積モデルである。
この断面積を波動散乱から得た厳密断面積だと主張しない。

高さ別の基準係数を使う場合、選んだ基準方向の断面積を `Cref` とし、

```
sigma_ref(h) = n(h) * Cref
sigma(h,k)  = sigma_ref(h) * C(k) / Cref
```

とする。方向別の光学的厚さを決める操作であって、NF 自体の条件依存を消す操作ではない。
`directional_coefficient()` がこの換算に対応する。基準方向の定義・端点の扱いはシーンに記録する。

## 3. Jakob 2010 への対応

論文の単一粒子の `p` と、RTE の `f_p` は同じ定義ではない。
Expanded Technical Report §3, Eq.(3), (6)--(10) は、粒子の配向分布から消散係数と inscattering kernel を作り、
出射側の散乱係数を外に出した `f_p` を導入する。

固定配向に特殊化すると、今回の記号では

```
f_p(ki -> ko) = K(ki,ko) / sigma_s(ko)
               = Csca(ki) / Csca(ko) * P(ko | ki)
```

となる。この係数比を掛けた因子を単一粒子の正規化済み PDF と混同しない。
入射方向についての正規化は、元の kernel と方向係数が物理的なバランス関係を満たすときに成立する。
一般の近似 NF に係数比を掛けるだけで相反性が自動的に強制されるわけではない。

このため `anisotropic.hpp` は物理的な `kernel` と、実際に生成した方向の `proposal_pdf` を分離する。
相反性を仮定して常に throughput=1 にしたり、NF を勝手に対称化・再正規化したりしない。
これは教師データ比較の要求ではなく、採用する学習済み関数を RTE に接続する際の規約である。

一次資料:
https://www.cs.cornell.edu/projects/diffusion-sg10/
https://www.cs.cornell.edu/projects/diffusion-sg10/diffusion-sg10-tr.pdf

## 4. カメラ側の逆追跡

カメラ側から散乱点へ進むレイ方向を `d`、そこから次の頂点または太陽へ向かう方向を `w` とすると、
物理的な散乱の問い合わせは

```
ki = -w
ko = -d
```

である。衝突位置のサンプル密度に含まれる消散係数は `sigma_t(x,-d)`。
正規化された任意の proposal `q(w)` を使った角度重みは

```
K(x; -w, -d) / (sigma_t(x,-d) * q(w))
```

となる。`camera_angular_weight()` はこの代数式を実装する。
例えば `q(w)=P(w|d)` は NF から直接生成できる proposal だが、物理 kernel と等しいと仮定しない。
このモデルの断面積は方向反転に対して偶関数だが、NF の符号付き条件を絶対値に置き換えない。

Rayleigh と雨滴を混ぜる場合は

```
sigma_t(x,k) = sigma_R(x,lambda) + n_r(x)*Cext(k,lambda)
K(x;ki,ko)   = sigma_R(x,lambda)*p_R(dot(ki,ko))
                + n_r(x)*Csca(ki,lambda)*P(ko|ki,lambda)
```

を用いる。成分を選んで方向を生成した場合も、MIS にはその生成法に対応する PDF を使う。

NEE の shadow ray `w` の減衰は、光が進んでくる `-w` の消散係数で評価する。
Null collision / ratio tracking に必要なのは係数のみであり、`cross_sections()` は MLP/HG/RQS を実行しない。

## 5. 座標・境界

NF のローカル z 軸は `ki`、x 軸は粒子軸の `ki` 直交面への射影、y 軸は `z cross x`。
鏡映対称性を保つ円周 RQS と、そのシームの結合微分を維持する。

NN の条件範囲は [380,830] nm × [-1,1] だが、物理 g/断面積表は
[-0.9999862922466825,+0.9999862922466825] までであり、極方向の小さい endcap を含まない。
現在の推論 API は範囲外を明示的に拒否する。端点の外挿、clamp、HG への代替は実装していない。
雨域を含む全方向レンダリングに接続する前に、この endcap の物理量の扱いを明示的に確定する。

## 6. 今回含まれないもの

M3 の自由飛行・NEE・PRD・LaunchParams・GUI は変更していない。
雨域の形、密度プロファイル、配向の空間変化、両媒質の混合サンプリング、
保存メタデータの拡張、実レンダリングへの接続は次の段階である。
本成果物はそれらを置き換える debug renderer ではなく、そのまま利用するための独立した推論層である。

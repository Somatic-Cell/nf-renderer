# M4 spherical Mie 1 mm backend — 実 LUT 統合

## 対象と実施範囲

ベース: `Somatic-Cell/nf-renderer` commit `c94d3cd9afe508e9fe9cbb720d5a42a4a923e385`（M4 rain transport を含む版）。近時の `flow.hpp` の `near` マクロ衝突修正は本差分の対象外。手元に修正済みならそのまま保持すること。

**統合コードを作成し、CPU の Mie 実データローダーと sampling / evaluation は検証済み。Windows/CUDA/OptiX によるレンダラ本体のビルドと画像描画は未検証。** NEE、既存 NF、Rayleigh の物理的な妥当性がここで追加検証されたとみなさない。

### 追加点

- `rain.phaseModel = "nf_anisotropic"` (既定値、旧 scene 互換) / `"sphere_mie_1mm"`。
- `rain.crossSectionMode = "geometric_matched"` / `"physical_mie"`。
- `rain.sphereMieTableFile = "assets/m4_mie/mie_sphere_1mm.bin"`、`rain.sphereMieManifestFile = "assets/m4_mie/mie_sphere_1mm.manifest.json"`。
- `MIE1LUT v1` loader: magic / size / shape / λ/θ 軸 / 確率質量 / alias / g / Q を検査。実 manifest の SHA-256 / size / λ/Q/g と照合する。未完成や改変は拒否。
- Mie LUT を一つの GPU アロケーションへ移送。旧 NF との混用なし、球形モードでは NF 重みをアップロードしない。OptiX direct callable は既存 slot 11 を使い、ファイルと entry function を切り替える。
- 雨滴 `free_flight()` / `transmittance()` の物理断面積照会は backend に応じて方向依存 Aproj または等方 sphere Cext に分岐。Null collision / ratio tracking 候補では重い位相関数を呼ばない。
- 散乱イベントでは球形に対し `p_Mie(dot(d,w),λ)` を評価 / sample。`value = (Csca/Cext) p`、`pdf=p`、`weight=Csca/Cext`。非吸収では1に近い。光源 NEE・MIS と continuation は元の M4 rain shader の経路を使用する。
- Mie デバイス演算の FP64 命令チェックを新しい PTX audit ターゲットへ追加。コードの実コンパイルは実機側で確認する必要がある。
- `linear.json` に選択された `phaseModel`, `crossSectionMode`, LUT SHA256 を保存（以前の `rain.modelSha256` フィールド名は NF との互換性のため維持）。

## セットアップと最初の実行

成果物をリポジトリ**外**に展開し、展開先から:

```powershell
python .\apply_m4_mie_backend.py --repo "C:\Users\sy415\workspace\pf-renderer" --check
python .\apply_m4_mie_backend.py --repo "C:\Users\sy415\workspace\pf-renderer" --apply
```

既存ファイルの Git blob SHA1 が公開ベースと異なる場合は、他の編集を破壊しないため適用を拒否する。差分は `m4_mie_backend.patch` にも記録。

```powershell
cd C:\Users\sy415\workspace\pf-renderer
cmake --build build --config Release --target NeuralPFRenderer
```

**`flow.hpp` の `near` マクロ修正がまだ入っていなければ先に対応する。** M4 の現在のバージョンは GPU の OptiX コンパイル等により追加エラーが発生する可能性がある。ログを保存して調整する。

起動例:

```powershell
.\build\bin\Release\NeuralPFRenderer.exe m4_mie_sky_geometric.json
.\build\bin\Release\NeuralPFRenderer.exe m4_mie_sky_physical.json
.\build\bin\Release\NeuralPFRenderer.exe m4_nf_sky_comparison.json
```

すべての比較シーンは露出倍率=100（表示のみ）、初期密度 30 個/m³、カメラ・太陽・乱数 seed 等同一。NN 用 `model.bin` は元の NF シーンのみ必要。球形シーンは Mie `.bin` と `.manifest.json` を読み込む。

## 検証

```powershell
cmake -S tests/m4_mie -B build-m4-mie
cmake --build build-m4-mie --config Release
ctest --test-dir build-m4-mie -C Release --output-on-failure
```

受領データの SHA-256: `e1d85bba3ef637444ec686455865b3b2235e3870d722a9e0776cf37230099332`。91 λ（380–830nm, 5nm）、15450 θ セル、16,934,692 byte。masses の行合計の最大誤差 `3.8e-9`、alias 復元誤差 `4.1e-10`。GNU C++17 CPU 81,920 サンプル実行成功、Clang ASan/UBSan 成功。FP32 の方向丸めにより極狭ビンで別の隣接ビンとして再評価される頻度は数十/81920 程度。`sample()` は再評価 PDF を返すが、全方向の統計・画像収束と GPU での誤差検証は別途必要。

**重要:** 残る数値検証は Mie を直接計算した高解像度参照に対する 125–130°, 135–145° の局所値と、91/451 λ、preview/final θ グリッドの比較。正規化と g の一致だけでは干渉ピークの保存精度を証明できない。

## 副虹を強調したい場合の密度設定

`numberDensityAtMinY = n` (個/m³)。雨域を横断する典型的な光学深さ `τ = n * Cext(λ) * L`。今回の550 nmの Qext≈2.004 と半径1mmから、`Cext≈6.30e-6 m²`。たとえば雨域を1km進む場合 n=30→τ≈0.189、n=100→τ≈0.63、n=300→τ≈1.89、n=1000→τ≈6.30。デフォルトの雨域は複数km、n=100 でもτは数程度になる。副虹を明瞭にしたければ、密度だけでなく雨域の経路長、Rayleigh背景、露出、太陽方向を合わせて調整する。nを上げ過ぎると多重散乱でコントラストが消える。

全データを非吸収物理 Mie として評価したい場合 `physical_mie`。既存 NF と断面積仮定を揃えた比較には `geometric_matched`。**どちらも `phase` の値を定数倍率で上げる操作ではなく、位相関数の積分は1のまま**。`sunScale` や exposure は虹だけを選択的に強めない。副虹だけを人為的に増幅する表示は物理比較とは別扱いにする。

## 比較上の限界

両粒子は一方が厳密な Lorenz–Mie と一方が Sadeghi 2012 の近似 GO/干渉/局所回折 + NF であり、ソルバ固有の誤差も混在する。したがって、レンダリングの全差を非球形性だけに帰属させない。`physical_mie` は Qext≈2 で `geometric_matched` の約2倍の光学的厚さになり、**同じ数密度での画像を単純な形状だけの差とみなさない**。

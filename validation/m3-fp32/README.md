# M3 FP32 検証記録

この記録は、配布するソースで実際に行った CPU テストと CUDA コンパイルです。
Windows アプリ全体のリンク、OptiX module のロード、GPU 実行、画像、FPS は未検証です。

## 実施した検証

| 項目 | 結果 |
|---|---|
| CMake Release / C++17 の CPU 回帰 | 12 / 12 成功 |
| 統計検査 | 12 区間・波長条件で delta / ratio を各 65536 試行 |
| Delta | 無衝突率、8 区間の衝突確率・CDF を独立参照と比較 |
| Ratio（RR 無効） | 平均、二次モーメント、Poisson 候補数を独立参照と比較 |
| Ratio（既定 RR 有効） | 65536 試行。RR 342178 回発動、25007 試行が RR で終了。期待値との一致を確認 |
| 誤りを検出する試験 | T の二重乗算、ratio の hard cutoff の誤りを検出 |
| 球の幾何 | 126 接線ケースで hit/miss 不一致 0、35 球面方向の位置表現に成功 |
| 位相関数・area PDF・UV 接線 | 極小・巨大の入力と従来の回帰を含め成功 |
| シーン移行 | 物理・他のシーン項目を保持、入力不変、既存出力と同一パスを拒否 |
| 実 CUDA コンパイル | OptiX-IR 14 本、出力 kernel の PTX 2 本に成功 |
| 追加 PTX 出力 | 同じ OptiX shader 14 本も PTX へコンパイル成功 |
| M3 が使用するプログラムの FP64 | 選択される 11 entry の `.f64` 命令・宣言・変換は 0 |

統計判定は原則 6 標準誤差と明示した FP32 許容差 3e-5 です。独立参照は long double の Cartesian
幾何と adaptive Simpson 積分を使い、実行時の FP32 clip や追跡ソルバを参照計算に流用しません。
これは有限試行の回帰試験であり、全シーン・全パラメータの数学的保証ではありません。

CUDA の条件は nvcc 13.0.88、OptiX 9 の実ヘッダー、compute_86 / sm_86、C++20、RDC、line info です。
API stub や仮の preinclude、`--use_fast_math` は使っていません。
CPU と CUDA の対象ソースについて、検査の前後に SHA-256 の変化がないことを確認しています。

## FP64 検査の正確な範囲

`m3_fp32_ptx_scope.json` に各 entry 名と PTX の hash を記録しています。
対象は spectral raygen、メッシュの closest-hit / any-hit、shadow、M3 の miss、
pinhole / thin-lens の両レンズ callable、tone-map と出力 copy kernel の 11 entry です。
M1/M2/M3 は既存 `noEnvMap_spectral` miss を選択し、M3 の太陽・大気用 miss helper を使います。
M3 の BSDF と光源サンプリングは inline で、旧 BSDF / light callable は呼びません。

別モード用の旧 VDB / NF / LUT / environment と旧 BRDF / light callable には FP64 が残っています。
全 16 TU の PTX をまとめて検索すると、それらに由来する `.f64` は 483 箇所です。
`m3_fp32_shader_compile.json` の `all_fp32_ptx: false` はこの全体検索の結果で、
M3 用の範囲検査 `all_selected_entries_fp32: true` と区別してください。
既存の別モードを一括で書き換えた、という主張はしていません。

PTX は OptiX-IR と同じソース・コンパイラ条件から別途生成した検査用出力です。
GPU で OptiX JIT が生成した SASS を検査したものではありません。
コンパイラが違う環境で最終命令が同じになると保証するものでもありません。

## 同梱ファイル

- `m3_fp32_cpu_checks.json` / `.log` / `m3_fp32_cpu_junit.xml`: CPU コマンド・結果・入力 hash。
- `m3_fp32_shader_compile.json` / `.log`: 全 30 コンパイルのコマンド・結果・入力と出力 hash。
- `m3_fp32_ptx_scope.json`: M3 が選択する entry と、別モードに残る FP64 の区別。
- `migration_check.json`: シーン変換の実行確認。
- `m3_fp32_base_verification.json`: GitHub 基準 tree の blob と照合した記録。
- `m3_fp32_patch_verification.json`: 2 本の差分の適用と、適用後ソースの byte 一致。

ソースの再現には ZIP の overlay と既存 SDK を使います。ビルド済み GPU バイナリは配布していません。
CPU の再実行方法は `docs/M3_RAYLEIGH_JA.md` に記載しています。

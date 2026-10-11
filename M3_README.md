# nf-renderer M3: Rayleigh 大気

対象 commit: `6e4794fb50fe3d9eafaacd78e5fe0b6e2098a103`

ZIP 内のファイルをリポジトリの同じ相対パスへ配置し、既存の build を再構成・ビルドしてください。
現在の OptiX レンダラを使用します。`cuda_driver_wrapper` は不要です。

```powershell
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release
.\build\bin\Release\NeuralPFRenderer.exe m3_rayleigh_sky.json
```

空だけのシーンで確認した後、`m3_emerald_day.json`、`m3_emerald_overview.json`、
`m3_emerald_sunset.json` を指定してください。`m3_emerald_vacuum.json` は球面地表を維持し、
大気の散乱係数だけを 0 にした対照です。

主な追加は、球状指数 Rayleigh 大気、多重散乱、太陽 NEE、地球遮蔽と Lambert 地表です。
1 パス 1 波長を維持します。光学的厚さと自由飛行は数値参照の積分・逆 CDF です。
大気 LUT、tracking、ニューラル重みの読み込みは使いません。

面光源 NEE は既定で大気の透過率を含みます。`rayleigh.areaNeeTransmittance=false` で
省略できますが、MIS 設定に依存する近似になるため、基準の検査には true を使ってください。
媒質の散乱点で行う NEE は太陽のみです。

適用、モデルと数値誤差、全パラメータ、診断、CPU 検査は [docs/M3_RAYLEIGH_JA.md](docs/M3_RAYLEIGH_JA.md)。
変更一覧は `M3_MANIFEST.json`、差分は `m3-rayleigh.patch`、実施した検証の記録は `validation/m3/` です。
Windows 全体のビルド・GPU 描画・FPS はこの環境では未検証です。

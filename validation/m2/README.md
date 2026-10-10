# M2 の検証記録

`summary.json` が検証範囲の一覧、`base_verification.json` が元の GitHub commit と
変更対象 16 ファイルの blob ID 照合結果。

- `geometry_audit.json`: 実 Day FBX を、production の `orca_scene.hpp` と
  `orca_material.hpp` を使って読み込んだ全 geometry・座標の監査。
- `material_audit.json`: 実 Day 全 220 材質の参照・定数・必要 UV と集計。
- `asset_sources.json`: 公式 ZIP の DDS 全ヘッダーに基づく形式・寸法・保持容量。
- `glass_alpha_audit.json`: 取得した Glass BaseColor 4 枚の実 alpha 集計。
- `cpu_ctest.log`: BSDF・スペクトル・シーン・alpha/MIS の 4 CTest。
- `shader_compile.json` / `.log`: 14 OptiX IR + 2 PTX の実 NVCC コンパイル。
  検証環境の絶対パスを記録している。コンパイル済み出力はこの ZIP に同梱しない。

`inspect_geometry.cpp` と `audit_materials.cpp` は監査に使用した C++ ソース。
Assimp commit `3e8673b14d2a209852dd1a1633464fe03c31746d`、nlohmann JSON、
このリポジトリの `include/` が必要。Windows renderer を模擬する stub は使用していない。
監査コードは通常の renderer target に追加していない。

例として、Assimp を別途 static build した Linux 環境では次のように実行できる。
`ASSIMP_SOURCE` と `ASSIMP_BUILD` は実際の依存配置へ置き換える。

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I include -I ext/json/single_include \
  -I ASSIMP_SOURCE/include -I ASSIMP_BUILD/include \
  validation/m2/inspect_geometry.cpp \
  ASSIMP_BUILD/lib/libassimp.a \
  ASSIMP_BUILD/contrib/zlib/libzlibstatic.a -pthread -o inspect_geometry
./inspect_geometry model/mesh/EmeraldSquare_v4_1/EmeraldSquare_Day.fbx
```

上記は CPU とソースレベルの検証である。
Windows アプリ全体のリンク・起動、全 DDS の DirectXTex decode、OptiX 実行、
画像の収束・見た目、速度、VRAM 使用量を確認した記録ではない。

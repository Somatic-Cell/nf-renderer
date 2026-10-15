# M4 NF: Windows の min/max マクロ衝突の修正

## 対象

`nf-renderer-M4-NF-inference-965a8f5.zip` を適用済みのプロジェクトに対する追加修正です。
M3 のコード、モデル資産、NN の構成、FP16 CoopVec / FP32 の計算仕様、RQS、PDF、輸送式は変更していません。

## 原因

添付ログの末尾では、`probe_optix.cpp` と `gpu_asset.cpp` に対して、関数形式マクロ `max` の引数不足が報告されています。
両ファイルには `std::numeric_limits<unsigned>::max()` があり、M4 の独立した CMake プロジェクトには `NOMINMAX` の定義がありませんでした。

OptiX 9 の `optix_stubs.h` は Windows 上で `Windows.h` を include します。
Windows の `min` / `max` マクロと C++ の同名関数が衝突すると、`std::max(...)` や
`std::numeric_limits<unsigned>::max()` がプリプロセッサで書き換えられます。
その結果、SDK のスタックサイズユーティリティ側も含め、複数の構文エラーが連鎖します。
スタックサイズの数値設定、FP16 演算、CUDA の精度設定を変える修正ではありません。

## 変更するファイル

| 相対パス | 修正 |
|---|---|
| `tests/m4_nf/CMakeLists.txt` | Windows 時の `m4_nf_probe_gpu` に `NOMINMAX` を定義 |
| `src/m4_nf/probe_optix.cpp` | `numeric_limits` の呼び出しを括弧付きに変更 |
| `src/m4_nf/gpu_asset.cpp` | 同上 |

`tests/m4_nf/CMakeLists.txt` は独立した M4 単体ビルド用です。**リポジトリ直下の `CMakeLists.txt` ではありません。**

CMake の主な追加部分は次のとおりです。`m4_nf_probe_gpu` を作成した後に置きます。

```cmake
if(WIN32)
  target_compile_definitions(m4_nf_probe_gpu PRIVATE NOMINMAX)
endif()
```

コンパイラ定義として与えるため、Windows / OptiX のヘッダーを処理する前に有効になります。
`PRIVATE` なので設定を他のターゲットや依存ライブラリへ伝播させません。

C++ 側では次のように変更しています。

```cpp
// Before
std::numeric_limits<unsigned>::max()

// After
(std::numeric_limits<unsigned>::max)()
```

C++ 側の括弧だけでは、SDK ヘッダー内の衝突までは解消しません。CMake の修正も必ず反映してください。
SDK ファイルそのものや `.vcxproj` の手修正は不要です。

## 適用

ZIP 内の `tests/` と `src/` を、リポジトリの同名ディレクトリへマージし、上記 3 ファイルを差し替えてください。
既存のディレクトリ全体を削除・置換しないでください。

手元で同じファイルを編集している場合は、同梱の `m4-nominmax-fix.patch` を確認して変更を統合してください。
リポジトリ直下へパッチをコピーした場合は、次の方法でも適用できます。

```powershell
git apply --check m4-nominmax-fix.patch
git apply m4-nominmax-fix.patch
```

ファイル上書きとパッチ適用は、どちらか一方だけ行います。

## 再ビルド

**クリーンビルドや `build-m4` の削除は不要です。** リポジトリ直下で実行します。

```powershell
cmake -S tests/m4_nf -B build-m4 -DM4_BUILD_OPTIX=ON
cmake --build build-m4 --config Release
ctest --test-dir build-m4 -C Release --output-on-failure
```

既に設定済みの `OPTIX_ROOT` や CUDA パスは CMake のキャッシュを利用します。
新しく構成する場合だけ、実際の SDK の場所を追加してください。

```powershell
cmake -S tests/m4_nf -B build-m4 -DM4_BUILD_OPTIX=ON `
  -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
```

これは M4 単体ビルドなので Assimp、NanoVDB、Emerald のアセットは読み込みません。
ホストのコンパイル設定変更により GPU probe の C++ ファイルは再コンパイルされます。
CUDA のカスタムコマンド、モデル、シェーダーは変更していません。

## 実施した確認と限界

- GCC 14.2 / Clang で、Windows と同じ形の `min` / `max` マクロを定義する**最小再現プログラム**をコンパイル。
  無対策では失敗し、`NOMINMAX` を先に定義した場合はコンパイル・実行に成功しました。
- 括弧付きの呼び出しが、同じマクロ定義の下でもコンパイル・実行できることを確認しました。
- Linux / GCC の Release 構成で M4 CPU 単体ビルドと CTest を再実行し、1 実行ファイル内の 6 群が成功しました。
- 前回の ZIP を新しく展開したものに対して `git apply --check` と実適用が成功し、適用結果が配布済みの 3 ファイルとバイト単位で一致しました。
- それ以外の元ファイルに変更がないことを照合しました。

**最小再現は Windows SDK や OptiX SDK 全体のビルドではありません。Windows / MSVC の GPU probe ビルド、NVCC の実行、実際の CoopVec 推論はこの環境では未実施です。**

結果・コマンド出力・最小再現ソースは `validation/m4_nominmax/` に収録しています。

## 参考資料

- Microsoft: max macro — https://learn.microsoft.com/en-us/windows/win32/multimedia/max
- Microsoft: XMMax remarks / NOMINMAX — https://learn.microsoft.com/en-us/windows/win32/dxmath/xmmax-template
- NVIDIA: OptiX 9.0 optix_stubs.h — https://raw.githubusercontent.com/NVIDIA/optix-dev/v9.0.0/include/optix_stubs.h
- CMake: target_compile_definitions — https://cmake.org/cmake/help/latest/command/target_compile_definitions.html

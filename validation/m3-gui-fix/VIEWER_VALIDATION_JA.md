# viewer のビルド修正と検証

## 添付ログの原因

以前の `viewer/CMakeLists.txt` には MSVC の `/utf-8` がありませんでした。
レンダラ本体に付けたオプションは、別の `viewer` ターゲットには伝わりません。
また、ルートで `add_subdirectory(viewer)` の後に追加したディレクトリの
コンパイルオプションも、評価済みの子ディレクトリには遡って適用されません。

`glfw_window.hpp` と `glfw_window.cpp` は BOM なし UTF-8 です。
日本語コメントのバイト列を CP932 の先行バイト規則で走査すると、コメント末尾で
LF が取り込まれ、次の宣言がコメントに入る箇所があります。その改行数の減少を
加味すると、添付ログの 15 箇所で診断行番号が正確に一致します。
元ファイルの SHA-256、行番号の対応、走査方法は
`viewer_encoding_diagnosis.json` に記録しています。

これはソースとログのバイトレベルの整合性確認であり、MSVC を実行した結果ではありません。
修正では `viewer` 自体へ `/utf-8` を明示し、日本語コメントは維持しています。

## 併せて修正した C++ の問題

- `std::shared_ptr` に必要な `<memory>` を直接 include しました。
- `helper_math.h` の単項マイナスは `float3&` を受け取るため、一時値の
  `-normalize(...)` を渡す記述を `-1.0f * normalize(...)` に変更しました。
  カメラ方向の計算内容は同じです。
- `getCameraFramePtr()` は所有する `CameraFrame` への参照を返すようにしました。
  既存の唯一の呼び出し元は `CameraFrame&` で受け取っているため、従来の値返しは
  標準 C++ ではコンパイルできませんでした。公開メソッド名は維持しています。

## 実行した検証

1. 実際の `viewer/glfw_window.cpp` 全体を GCC、C++17、`-O2 -c` でコンパイルしました。
   終了コードは 0 で、警告・エラーはありません。コマンド、コンパイラ情報、
   入力ファイルのハッシュは `viewer_gcc_compile.json`、出力は
   `viewer_gcc_compile.log` にあります。
2. GLFW と ImGui はリポジトリの submodule に記録された commit の実ヘッダを使用しました。
   CUDA は実際の CUDA 13 ヘッダ、OpenGL は固定した NVIDIA/libglvnd の実ヘッダを使用しました。
   代用の型定義や関数宣言は作成していません。
3. 独立した CMake の generator expression 検査で、MSVC 条件の `viewer` の
   コンパイルオプションが `/utf-8`、GNU 条件では空になることを確認しました。
   `viewer_cmake_options_probe.json` に検査用 CMakeLists と結果を収録しています。
   これは条件分岐の検査であり、MSVC や Visual Studio の実ビルドではありません。

この環境では Windows の全アプリケーションのリンク、GUI の起動、GPU 上での
レンダリングは検証していません。

## 一次資料

- Microsoft: `/utf-8` がない場合、BOM のないソースは current user code page で解釈されます。
  https://learn.microsoft.com/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8
- CMake: `add_subdirectory` は呼び出し位置で子ディレクトリを直ちに評価します。
  https://cmake.org/cmake/help/latest/command/add_subdirectory.html
- CMake: ターゲットの `COMPILE_OPTIONS` は作成時にディレクトリのプロパティから初期化されます。
  https://cmake.org/cmake/help/latest/prop_tgt/COMPILE_OPTIONS.html

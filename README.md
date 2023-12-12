# CS 6120 Final Project: Polyhedral Compilation

## Setup

Be sure to initialize submodules when cloning:

```sh
git clone --recurse-submodules <repository>
```

Build LLVM, MLIR, Clang, and Polygeist according to
[Polygeist's documentation](https://polygeist.llvm.org/Installation/).
If you get link errors pertaining to `stdc++fs`, try commenting out
[this line](./extern/Polygeist/lib/polygeist/Passes/CMakeLists.txt#L65).

You'll also need [isl](https://libisl.sourceforge.io/). On a Mac, it can be
installed via Homebrew:

```sh
brew install isl
```

Now build the project as follows:

```sh
mkdir build && cd build
cmake .. -DMLIR_DIR=$PWD/../extern/Polygeist/llvm-project/build/lib/cmake/mlir
cmake --build .
```

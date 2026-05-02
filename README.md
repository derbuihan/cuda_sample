# cuda_sample

Numba-CUDAで小さなPyTorch風Tensorライブラリを作りながら、CUDAと自動微分の低レイヤーを学ぶための実験リポジトリです。

目的は、既存の機械学習ライブラリをただ使うのではなく、その下にある仕組みを小さく作って理解することです。Tensor、CUDA kernel、行列演算、自動微分、optimizerを少しずつ実装し、最終的には自作kernelでMNISTを学習するところを目指します。

## コンセプト

このプロジェクトでは、Python側でPyTorch風のAPIを作り、重い数値計算はNumba-CUDA kernelとして実装します。

```text
Python API
  ↓
Tensor / autograd / optimizer
  ↓
Numba-CUDA kernel
  ↓
PTX / CUDA driver
  ↓
NVIDIA GPU
```

最初から汎用ライブラリを目指すのではなく、学習しやすさを優先します。

- TensorはCUDA中心に扱う
- CPUに戻すのは表示やデバッグのときだけ
- まずは正しさを優先し、高速化は後から考える
- PyTorchの表面だけでなく、下にある計算グラフやkernel実行を意識する

## 構成

```text
src/
  main.py      # 実験用エントリポイント
  tensor.py    # Tensorとautogradの実装
  kernels.py   # Numba-CUDA kernel

samples/
  main.py      # PyTorchでの比較用MNISTサンプル
  main.cu      # CUDA C++サンプル
  Makefile
```

## 実行

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 -m src.main
```

CUDA C++サンプルは `samples/` 配下で実行します。

```bash
cd samples
make run
```

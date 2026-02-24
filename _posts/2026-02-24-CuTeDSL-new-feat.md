---
title: 'CUTLASS：我们走了一些弯路'
date: 2026-02-26
permalink: /posts/2026/02/CuTeDSL-new-feat/
tags: 
  - GPU
  - AI
---

去年我在 [英伟达反对英伟达](https://enigmahuang.me/posts/2025/05/NV-against-NV/) 一文里面吐槽过 CUTLASS 的易用性。大半年后的今天，我们看到了一些有意思的变化。

## （不）易用性

去年下半年 CuTeDSL 上线后立即收到了大量的关注，连 Flash Attention 4 都是用 CuTeDSL 实现的。和 CUTLASS C++ 相比，CuTeDSL 终于摆脱了 C++ 模板地狱，导致各位搓 kernel 的兄弟再也没有『等编译』这个合理摸鱼的理由了。这解决了 CUTLASS C++ 的一大痛点，但是另一大痛点，也是 CUTLASS 3.0 设计最突出的特点和优点，CuTe layout, 其易用性问题依然没有解决。以 v4.4 的 Blackwell FP16 GEMM [教程代码](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) 为例。

大部分 CUDA kernel 或者说几乎所有需要用 Tensor Core 的 kernel, 主要做三件事：1. 划分全局数据，把自己需要使用的数据从 gmem 读取到 smem 或者 reg 里 (prologue)；2. 用 smem / reg 里面的数据进行计算 (compute/mainloop)；3. 把计算得到的结果写回全局 gmem (epilogue). 选用 Blackwell 代码而非 Hopper 代码是因为我觉得 Blackwell 拆分 mainloop 和 epilogue 比 Hopper 不拆分的更合理和更好理解。大部分情况下，一个线程组 (CTA) 从 gmem 读写的数据都可以视作一个矩阵的子块。 Triton, cuTile 等 tile-based / array-based 编程语言基本上也是按这样来划分函数功能组别的。那么在 CuTeDSL 里面呢？以上面的代码为例，和输入矩阵 $A$ 直接相关的局部变量，分别有：`mA_mkl`, `gA`, `sA`, `tCgA`, `tCrA`, `tAsA`. 而和输出矩阵 $C$ 直接相关的局部变量，分别有：`mC_mnl`, `gC`, `tCtAcc`, `tCtAcc_epi`, `gC_epi`, `tDtC`, `tDgC`, `tCrAcc`, `tCrC`. 啊，为什么 `tCgA` 和 $A$ 相关但是不是和 $C$ 相关？`tCgA` 和 `tCrA` 的区别是什么？为了方便理解，我画了一张变量依赖关系图：

![SM100-GEMM-dep-1](http://enigmahuang.github.io/files/CuTeDLS-new-feat/SM100-GEMM-dep-1.png)

实际上，CUTLASS 里面有大量形如的 `tXlY` 的变量命名，其中 `X` 和 `Y` 和划分对象有关，`l` 和存储位置有关，但是 CUTLASS 的文档里从来没有解释过应该如何解读这样的变量名。让我们暂时忍一下种类繁多的中间变量，进一步了解一下一些核心函数。比如最核心的变量之一，`tiled_mma = cute.make_tiled_mma(op)`. 查看 [CUTLASS 文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html) 可知这个函数返回了一个 `TiledMma` 类，这个类派生自 `MmaAtom` 类, 包含 `get_slice`, `make_fragment_{A,B,C}` 等常见且核心的操作。然而官方文档里，这两个类的成员和方法函数没有任何说明：没有说明输入参数有什么要求，没有说明成员的功能和期待的输出是什么、满足什么条件。

![CUTLASS-doc-1](http://enigmahuang.github.io/files/CuTeDLS-new-feat/CUTLASS-doc-1.png)

![CUTLASS-doc-2](http://enigmahuang.github.io/files/CuTeDLS-new-feat/CUTLASS-doc-2.png)

且慢， CuTeDSL 提供了一些调试用的打印功能，可以打印出一些变量信息帮助理解。比如下面这一段代码（148-155行）：

```python
# (MMA, MMA_M, MMA_K)
tCrA = tiled_mma.make_fragment_A(sA)
# (MMA, MMA_N, MMA_K)
tCrB = tiled_mma.make_fragment_B(sB)
# (MMA, MMA_M, MMA_N)
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
# (MMA, MMA_M, MMA_N)
tCtAcc = tiled_mma.make_fragment_C(acc_shape)
```

如果我们打印出 `tCrA` 和 `tCrB` 的信息，可以看到它们的形状是 `(1,1,4,4)`. 其中最后一个 4 是 `MMA_K = mma_tiler_mnk[2] / mma_inst_shape_mnk[2]`, 即在一个分块里面 K 维度方向需要使用多少次 MMA atom; 倒数第二个 4 其实是代码里设定的 `ab_stage = 4`, 那么前面的两个 1 呢？注释里标注的可是一个三维元组，不是一个四维元组。把 `tCtAcc` 也打印出来，可以看到它的形状为 `((128,256),1,1)`. 这倒是好理解一点：`(128,256)` 就是 $C$ 矩阵的分块大小，对应注释里的 `MMA`. 但为什么和上面的 `MMA` 不同？

写到这里我不禁要再次感叹：世界上到底有多少聪明的大脑能完全搞懂 CuTe 背后到底在干什么？

## 『我们在探索中走了一些弯路』

CUTLASS v4.4 带来了一些 CuTeDSL 的实验特性：数据划分逻辑简化、自动生成 TMA 描述符、无 fragment 编程模式、pipeline 接口简化等。这个 [新样例代码](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py) 展现了大部分的新特性。这个新的样例代码写了大量的注释来解释这些操作，很值得一读，个人觉得比此前所有的样例代码都更有价值。在仔细阅读这个样例代码之前，不妨先看看下面这张变量关系依赖图：

![SM100-GEMM-dep-2](http://enigmahuang.github.io/files/CuTeDLS-new-feat/SM100-GEMM-dep-2.png)

和上一张变量关系依赖图相比，可以看到新代码大幅减少了涉及的变量，而且 prologue 和 mainloop 部分的逻辑也大幅度简化了，以及数据流动的方向也用带箭头的虚线标记了出来。

对于 TMA 复制，新代码提供了一些非常方便的新接口：`make_smem_layout_{a,b}` 和 `make_tmem_layout_acc`. 和自动生成 TMA 描述符搭配，在 prologue 数据复制和 mainloop 计算这两个部分，代码写起来已经比较接近 Triton/cuTile 这些语言了，对降低程序员心智负担有极大的帮助。诚然，这些辅助函数背后还是有一些更复杂的操作，但是大部分时候程序员并不真的需要知道或者更改实际上的数据排布，编程框架就应该提供一个选项把这些复杂度隐藏起来。 Epilogue 部分仍然相对复杂，这主要是由于：1. 数据流动的阶段相对较多：要先从 tmem 复制到 reg, 在 reg 里面完成后处理，再复制到 smem, 最后用 TMA 写回 gmem; 2. 涉及到多次数据排布格式 (CuTe layout) 的更改。这个 epilogue 比教程代码里的实现更复杂，因为不是直接从 reg -> gmem, 而是经 smem 用 TMA 进行写入。但是如果和 [性能更好的样例代码](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_gemm.py) 里的 `epilogue_tma_store()` 实现对比，那也是大幅度简化了。考虑到 epilogue 的模式都比较固定，如果以后能有进一步的辅助函数，类似现有的用于 [dense_gemm_persistent.py](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py#L1006) 里的 [epilogue_tma_store](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/python/CuTeDSL/cutlass/utils/gemm/sm100.py#L157) 一样的辅助函数，那将更进一步降低复杂度。

一个顺理成章的问题是，如果 CuTeDSL 在未来提供更多的辅助函数和简化的接口，那么它和 Triton/cuTile 这些框架的区别是什么？目前看来，CuTeDSL 还可以手动控制 pipeline 深度，以及做一些更细颗粒度的调度和操作（比如 [overlap and double buffer accumulator](https://github.com/NVIDIA/cutlass/blob/3476ddb7bd6ca4161a0169103ceaa20ce0eb891f/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L368)）。Triton/cuTile 目前依赖编译器的 auto wrap specialization, 会在编译和运行过程中试图对操作自动进行切片和将不同操作的切片进行流水线拼接。这样的灵活性和最后的性能，在短期内可能还是会比手动控制的要稍微差一点。

总而言之，随着新语言特性和接口的引入，CuTeDSL 在解决易用性方面，又迈出了重要的一步。或许再过几年，随着编译器的发展，tile-based DSL 就能挑起大梁了呢？那就真的说不好是『好时代，来临了』还是坏时代来临了。

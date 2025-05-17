---
title: '英伟达反对英伟达'
date: 2025-05-17
permalink: /posts/2025/05/NV-against-NV/
tags: 
  - GPU
  - Hardware
  - AI
---

## DeepSeek 一声炮响

2025 年 2 月 26 日，DeepSeek 开源周第三天，DeepGEMM 代码开源发布。围绕 DeepSeek 开源代码的讨论有很多，最抓人眼球的是『DeepSeek 打破 CUDA 壁垒』、『DeepSeek 撕开英伟达垄断圈』一类的说法。稍微懂行的人都知道此类说法基本都是新闻学魅力时刻，严肃正经的反驳和讨论也已有很多。泥沙混流之中，还有一些有趣的东西值得捞起来看一看。

打开 [DeepGEMM 代码仓库](https://github.com/deepseek-ai/DeepGEMM)，README 文件第二段有这么一句话：

> While it leverages some concepts from CUTLASS and CuTe, it avoids heavy reliance on their templates or algebras.

这里面出现了两个名字：CUTLASS 和 CuTe. CUTLASS 的全称是 CUDA Templates for Linear Algebra Subroutines, 看起来和 Basic Linear Algebra Subroutine (BLAS) 颇为相似。实际上 CUTLASS 并不这么 Linear Algebra, 它的目的只有一个：搓出性能最好的 AI kernel, 主要是 GEMM. 传统的 cuBLAS 库也在与时俱进提供各种新的 GEMM 实现，然而受制于接口和发布的形式，cuBLAS 能支持的应用场景还是不如 CUTLASS. 如果需要定制开发 kernel, 比如 DeepGEMM 所需要的 FP8 fine-grained scaling, CUTLASS 理应是最好的选项。当然，Triton 也能定制开发 kernel, 甚至作为一门 DSL, Triton 用起来还更简单一点。但是除了 OpenAI 自己可以关起门来优化，有多少人能把 Triton kernel 写得和 CUTLASS kernel 一样快？至于 CuTe, 我们稍后再展开，只要知道它自 CUTLASS 3.0 开始是 CUTLASS 的一个核心组件。

那么问题来了：DeepGEMM 只是借用了一些 CUTLASS 和 CuTe 的组件，而没有用 CUTLASS 来实现，代价是什么？

## 走进新时代

2022 年，NVIDIA 发布了新一代 Hopper 架构的旗舰计算卡 H100. 比起前一代 A100, H100 的理论峰值性能，特别是 Tensor Core (TC) 性能，有了巨大的跃进：以 SXM 版本为对比，FP16/BF16/INT8 翻了三倍，从 312/312/624 T 变成了 939/939/1959 T, 同时引入了了高达 1959T 的 FP8 支持。硬件性能大跃进，软件自然也要跟上：H100 上引入了新的 WGMMA 指令集。对应地，CUTLASS 3.0 系列发布，TC 编程走进了新时代。

讨论何为『新时代』之前，有必要先回顾一下 good old times. 在 V100 和 A100 时期，TC 编程使用的指令是 WMMA/MMA. 这些指令的执行粒度都是单个 wrap, 需要 wrap 里面所有线程一起执行同一条指令。这种方式在别的 CUDA 库里面也有，比如 [CUB 的 WrapReduce](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html). 使用 WMMA/MMA 实现的 GEMM kernel, 比如[这个样例代码](https://github.com/wzsh/wmma_tensorcore_sample/blob/master/matrix_wmma/matrix_wmma/main.cu)，看着和传统的 GEMM kernel 还有几分相似，只是颗粒度变大了一点。一个熟练的 CUDA 码农，对着文档和样例学个一两天也就能上手了。

到了 Hopper 时代，好消息是传统的 MMA/WMMA 还可以继续用，坏消息是如果只用这些传统指令，代码最多最多只能跑到[大概 63% 的峰值性能](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)。而 H100 上面，每四个 wrap 组成了一个 quadrant, 里面有一个 wrap scheduler, 一个 TC, 512 个向量寄存器，和一些其他部件。WGMMA 对应的 wrap group 就是四个 wrap. 此外，还有全新的 tensor memory accelerator (TMA), 以及 TMA 支持的 memory swizzling. 更让人头痛的是，WGMMA 的 PTX 文档不仅难读懂，甚至还有错（见前一个链接）。往好了说，写文档的人自己都神情恍惚搞错了。更更吓人的是，为了让 TC 保持全速计算，WGMMA 出现了 wrap specialization, 即一个 wrap 专门负责搬运数据，其他 wrap 负责控制 TC 做计算。对应地，GPU 上的软件流水线出现了。

能够直接硬啃 PTX 然后手搓 WGMMA 的人实在是凤毛麟角。于是 CUTLASS 3.0 包装好了所有底层操作，变成了事实上的 H100 TC 编程文档和基本接口。CUTLASS 3.0 还带来了一个新的核心部件 CuTe, 用以 "describe and manipulate tensors of threads and data". 利用 CuTe 和 CUTLASS 官方实现和包装好的各种功能，开发者应该可以像搭积木一样组合和实现出各种功能的 kernel, 并且方便地调试各种模板参数来把性能推向极致。如此一来，可谓是以改兼振两难自解，我大明天下无敌啊！

平心而论，面对日益复杂的硬件架构和日益增加的功能需求，CUTLASS 3.0 所做的只是计算机科学两大方法论里的第一条：增加一层抽象以实现更多功能。但是同样平心而论，CUTLASS 3.0 着实有些抽象了。CuTe 的抽象可以很简洁地描述某些复杂的数据排布，但是 CuTe 这一套复杂的 layout algebra, 除了十多份官方文档，还有一篇英伟达员工在此之上另外写的一万七千字的[讲解文章](https://leimao.github.io/article/CuTe-Layout-Algebra/), 来帮助其他开发者学习 CuTe. 而 CUTLASS 本身，则是一个将 C++ 模板编程用到极致的典型。一个 kernel 有十几个 class 作为模板参数，VSCode language server 看了都崩溃，更别说人了。而这些大大小小的 class 和模板，也没有多少有文档，基本上只能靠看代码和看样例来推测其用法。哎，没事，写算子的朋友们只要苦一苦就好了，TC 要做的事情可就多了。

## 革命与自我革命

总体而言，H100 + CUTLASS 3.0 时代，硬件和软件的复杂性问题都开始浮出水面。传统的 (GP)GPU 的硬件设计，对应的并行编程模式是 SIMT, 依靠在大量轻量级的线程中进行切换和轮流执行以掩盖显存访问延迟和保持算术单元一直进行计算。这一范式在 H100 TC 上面已经被打破了，wrap specialization 和对应的 CUTLASS 里的多级软件流水线，和传统的 GPU 编程已经是完全不同的范式。始作俑者其无后乎，H100 打破了 V100 A100 的计算模式以获得性能上的大跃进，一堆 kernel 搓出来还没用多久，Blackwell 的 TC 架构和对应的编程模式又改了。虽然 Hopper 到 Blackwell 的变动远没有 Ampere 到 Hopper 的变动那么大，但是各种 kernel 还是得重新搓一次。那么 Blackwell 之后的 Rubin 呢？要做出多少牺牲、妥协、变化？

在软件层面，CUTLASS 团队显然也意识到了现在 CUTLASS 有多难用, DeepGEMM 没有用 CUTLASS 就是最好的例证。于是，CUTLASS 4.0 开始引入 Python DSL 的支持。[官方原话](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)如下：

> CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development. By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables faster iteration, easier prototyping, and a gentler learning curve for high-performance linear algebra on NVIDIA GPUs.

CUTLASS 有 Python DSL 是好事。毕竟如果延续现在这个学习难度和开发复杂度，加上 Blackwell 开始支持的 MX 数据类型和各种 sub-type 数据类型所需的操作，本就紧张的 CUTLASS 产能很可能会开始拖后腿。但是 CUTLASS Python DSL 似乎有些迟了，因为更激进的技术路线已经出现了：外有 [tile-lang](https://github.com/tile-ai/tilelang), 内有 GTC 25 上面公布的 [cuTile](https://www.linkedin.com/posts/brycelelbach_we-just-announced-cutile-a-tile-programming-activity-7308545706242306048-Lp_j/). 随着 cuTile 一起到来的还有 Tile IR, 这个东西看着可就有意思了：

![Tile IR](http://enigmahuang.github.io/files/NV-against-NV/TileIR.jpg)

在这张示意图里面，cuTile -> Tile IR 被称为 Tile path, 完全和 SIMT path 不同，也不走 PTX. cuTile 这种 "array-based paradigm" 里面为传统 CUDA 编程范式留下了多少容身之地，我们可以暂且打个问号。如果以后 cuTile 发展顺利，能在性能上达到一流水平，那么绝大部分需要给 NVIDIA GPU 编程的程序员就只需要学习 cuTile, 不需要学习传统的 CUDA 编程就可以满足工作需求了。换言之，旧时代的 CUDA 生态护城河，可能就会变成一道马奇诺防线了。

脑洞不妨再开得大一些。如果软件上的 DSL 大获成功，那么硬件设计上面会不会也日益 DSA 化？毕竟 NV 面临着每一代旗舰卡理论峰值性能起码翻一倍的压力，否则资本市场不会买账。半导体制程提升的红利越来越小，单个芯片的面积越来越大，砍掉和矩阵计算无关的单元的潜在收益会越来越高。比如，旗舰卡上的 ROP 和 TMU 单元明显是不需要的; 需要用 CUDA core 执行的计算也没有那么多，只要保留一部分来执行少量非矩阵运算就够了。如果以后真的变成了 DSA + DSL, 那么事情就变得有趣起来了。

最后是一个值得思考的问题：AMD 的 HIP 也好，没有后续的 zluda 项目也罢，还有各种国产显卡上号称『兼容 CUDA』的编程框架，面对 SIMT/WMMA/WGMMA/TCGEN5, 兼容哪些才算『兼容 CUDA』? 如果英伟达自己都不打算在最重要的产品上面兼容自己以前的编程模式，你们『兼容 CUDA』的意义又在哪里？

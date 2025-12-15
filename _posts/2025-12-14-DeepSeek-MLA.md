---
title: 'DeepSeek Multi-head Latent Attention 计算流程图'
date: 2025-12-14
permalink: /posts/2025/12/DeepSeek-MLA/
tags: 
  - AI
---

近日因工作需要去学习了一下大名鼎鼎的 DeepSeek Multi-head Latent Attention. MLA 的计算流程比标准的 GQA 要复杂不少，主要是处理低秩压缩和 RoPE. 搜了一些网上的资料，没有看到非常满意的计算流程图，因此自己动手画了两个。下面两张图都是针对推理的，不考虑反向传播计算梯度。第一张图是不使用矩阵吸收时的计算流程（常用于 prefill），第二张图是使用矩阵吸收时的计算流程（常用于 decode）。图中各个张量的大小都做了标注，维度名字大部分遵循 [DeepSeek V3 官方参数名](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json). 部分张量的 `head_dim` 和 `seq_len` 的位置可能会交换，不影响理解；矩阵乘法的内积维度未显式标出。

![MLA without matrix absorb](http://enigmahuang.github.io/files/MLA/MLA-no-absorb.jpg)

![MLA with matrix absorb](http://enigmahuang.github.io/files/MLA/MLA-with-absorb.jpg)

DSv3 参数：

- `hidden_size`: 7168
- `kv_lora_rank`: 512
- `q_lora_rank`: 1536
- `num_attention_heads`: 128 （图中写作 `num_heads`）
- `qk_nope_head_dim`: 128
- `qk_rope_head_dim`: 64
- `v_head_dim`: 128

我对照了 vLLM 里的代码执行路径，发现和我的流程图稍有不同，主要是对矩阵吸收的应用。其中，吸收/不吸收的共通路径在
[vllm/model_executor/layers/mla.py, L110-L173](https://github.com/vllm-project/vllm/blob/33278073d68940dcaff579ab2dc316700e1db87a/vllm/model_executor/layers/mla.py#L110-L173)
内部分支实现在
[vllm/v1/attention/backends/mla/common.py, L1927-L2114](https://github.com/vllm-project/vllm/blob/33278073d68940dcaff579ab2dc316700e1db87a/vllm/v1/attention/backends/mla/common.py#L1927-L2114)
共通路径部分大体和我流程图里的 Common part 相同，但是总是对 $c^Q_t$ 做 up projection ($q^C_t = c^Q_t \times W^{UQ}$). 实际上，vLLM 的矩阵吸收方式不是 $(c^Q_t \times W^{UQUK}) \times (c^K_t)^T$, 而是 $((c^Q_t \times W^{UQ}) \times W^{UK}) \times (c^K_t)^T$, 这样依然避免了显式计算出 up projection 以后的 $k^C_t = W^{UQ} \times (c^K_t)^T$. 带入 DSv3 的具体参数，可以计算得到 vLLM 这种方式所需的计算量更小。同样，在使用矩阵吸收的时候，会给 MQA 的输出乘以 $W^{UV}$, 最后回到共通部分再乘以 $W^{O}$ 得到最终输出。

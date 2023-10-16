---
title: '页面渲染测试'
date: 1970-01-01
permalink: /posts/1970/01/rendering-test/
tags:
---

### CJK Test

> 元年春王正月。
> 人間五十年、下天の内をくらぶれば、夢幻の如くなり。

### LaTeX Test

单行公式：$A v = \lambda B v $

多行公式：

$$
\begin{gathered}
A =
\begin{bmatrix}
L &  A &  P &  A &  C &  K \\
L & -A &  P & -A &  C & -K \\
L &  A &  P &  A & -C & -K \\
L & -A &  P & -A & -C &  K \\
L &  A & -P & -A &  C &  K \\
L & -A & -P &  A &  C & -K \\
\end{bmatrix}
\end{gathered}
$$

### Code Syntax Highlight Test

```cpp
#include <cstdio>
#include "utils.h"

template<typename KT, typename VT>
static void qsort_key_val(KT *key, VT *val, const int l, const int r)
{
    int i = l, j = r;
    const KT mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            KT tmp_key = key[i]; key[i] = key[j];  key[j] = tmp_key;
            VT tmp_val = val[i]; val[i] = val[j];  val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_key_val<KT, VT>(key, val, i, r);
    if (j > l) qsort_key_val<KT, VT>(key, val, l, j);
}
```
"""把 SageAttention 接进 transformers 的注意力接口。

transformers 的 `ALL_ATTENTION_FUNCTIONS` 里没有 sage，所以这里自己注册一个名字叫 "sage"
的注意力实现，然后节点就可以直接 `attn_implementation="sage"`。

策略是"能走就走、不能走就退"：
- 走 sageattn：CUDA + fp16/bf16 + causal + 没有 attention_mask + q_len 够长 + head_dim 对齐
- 其余一律退回 transformers 自带的 sdpa，保证任何情况下都不会跑坏

典型情况：
- 预填充（图片/提示词那一次前向）：无 mask、causal、q_len 够长 → **走 sage**
- 逐 token 解码：有 4D mask → 退回 sdpa（sage 对 q_len=1 本来也不划算）
- 视觉塔：显式 is_causal=False → 退回 sdpa

关于 MIN_Q_LEN（重要）
---------------------
sage 是 int8/fp8 量化内核，量化本身有固定开销；序列太短时这个开销吃不掉收益，
反而比 torch 自带的 sdpa 慢。本机（RTX 5090 / sm120 / torch 2.9.1+cu130）实测
单次 MHA、fp16、D=128、causal：

    S        sage           sdpa          加速比
    512      0.21 ms        0.10 ms       0.29x       稳亏
    1024     0.22 ms        0.10 ms       0.46x       稳亏
    1536     0.24~0.39 ms   0.16~0.19 ms  0.48~0.65x  稳亏
    2048     0.26 ms        0.25 ms       0.91~0.99x  持平
    2560     0.28~0.39 ms   0.34 ms       0.89~1.32x  不稳（出现过亏本轮次）
    3072     0.32 ms        0.43 ms       1.36~1.52x  稳赚
    4096     0.38~0.52 ms   0.69 ms       1.33~1.83x  稳赚
    8192     0.52 ms        2.35 ms       4.55x       稳赚

注意本机后台常驻 LDPlayer 模拟器、浏览器、NVIDIA Overlay（约 4.2 GB 显存），
sage 单次只有 0.2~0.5 ms，实测耗时呈双峰波动；sdpa 侧则很稳定。所以阈值取的是
**第一档"每个轮次都赚钱"的长度**，而不是平均交叉点——即 3072，而不是 2048~2560。

另一点：这还只是热循环单算子，真实前向额外要做一次 GQA 的 repeat_kv 拷贝
（8→32 头，2048×128 fp16 约 16MB，≈11us），实际只会更偏保守。

因此：
- 单图 VQA（默认 max_pixels=1280*28*28 → 约 1280 视觉词元）→ **不要用 sage，会慢 2~3 倍**
- 视频 / 多图 / 超长提示词（>3k 词元）→ sage 才有意义
- 阈值只在 2048~3072 这个窄区间影响结果，那里 sage 本来也就 1.0~1.4x，摊到整个
  8B 模型前向里更是微乎其微，所以宁可取保守值

想调整就设环境变量：

    QWEN3_VL_SAGE_MIN_Q_LEN=0      强制尽量走 sage（短序列会明显变慢）
    QWEN3_VL_SAGE_MIN_Q_LEN=8192   只在超长上下文才用
"""

import os

import torch

from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

NAME = "sage"

# 序列短于这个长度就不走 sage（见文件头实测表）。可用环境变量覆盖。
MIN_Q_LEN = int(os.environ.get("QWEN3_VL_SAGE_MIN_Q_LEN", "3072"))
HEAD_DIM_MULTIPLE = 64  # sage 对 head_dim 有对齐要求（Qwen3-VL 是 128，满足）

_available = None  # None=还没探测过
_warned_runtime = False
_warned_missing = False


def sageattn_available() -> bool:
    global _available, _warned_missing
    if _available is None:
        try:
            import sageattention  # noqa: F401

            _available = True
        except Exception as e:  # pragma: no cover - 取决于环境
            _available = False
            if not _warned_missing:
                _warned_missing = True
                print(f"[Qwen3_VL] 未检测到 sageattention，attention=sage 会全部退回 sdpa：{e}")
    return _available


def _fallback(module, query, key, value, attention_mask, dropout, scaling, is_causal, kwargs):
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )


def sage_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """签名和 transformers 的 sdpa_attention_forward 保持一致。"""
    global _warned_runtime

    # 和 sdpa 一样的 mask 裁剪逻辑，保证退路行为完全一致
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    def fb():
        return _fallback(
            module, query, key, value, attention_mask, dropout, scaling, is_causal, kwargs
        )

    # ---- 决定走不走 sage ----
    if not sageattn_available() or torch.jit.is_tracing():
        return fb()
    if attention_mask is not None or not is_causal or dropout:
        return fb()
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        return fb()
    if query.shape[2] < MIN_Q_LEN or query.shape[2] != key.shape[2]:
        return fb()
    if query.shape[-1] % HEAD_DIM_MULTIPLE:
        return fb()
    # sageattn 的默认缩放就是 head_dim**-0.5，别的缩放值不敢直接用，退回
    if scaling is not None:
        default_scale = query.shape[-1] ** -0.5
        if abs(float(scaling) - default_scale) > 1e-6:
            return fb()

    try:
        import sageattention

        k, v = key, value
        n_rep = getattr(module, "num_key_value_groups", 1) or 1
        if n_rep > 1:  # sageattn 要求 q/k/v 头数一致，先做 GQA 扩展
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        out = sageattention.sageattn(
            query,
            k,
            v,
            tensor_layout="HND",
            is_causal=True,
            output_dtype=query.dtype,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.transpose(1, 2).contiguous(), None
    except Exception as e:
        if not _warned_runtime:
            _warned_runtime = True
            print(f"[Qwen3_VL] sageattn 调用失败，本次及后续退回 sdpa：{e}")
        return fb()


def register() -> bool:
    """把 "sage" 注册进 transformers，失败不影响插件其它功能。

    要注册两张表，缺第二张会算错（不是报错）：
    1. `AttentionInterface`（注意力函数）—— 否则 `attn_implementation="sage"` 直接报错。
    2. `ALL_MASK_ATTENTION_FUNCTIONS`（mask 生成器）—— transformers 用它来决定
       "这个实现需不需要 4D 因果 mask"。我们的名字没在里面时它会走 early-exit
       （masking_utils.py:718 `if config._attn_implementation not in ...: return None`），
       于是**连 padding mask 一起被丢掉**。batch=1 时看不出问题（本来就没有 padding），
       但一旦有 padding（多图拼 batch / 变长序列），padding 位会被当成正常 token 参与注意力，
       结果是静默算错。挂上 sdpa 的 mask 生成器后：
       - 无 padding 的预填充：mask 仍为 None（`_ignore_causal_mask_sdpa` 会跳过），sage 照常生效
       - 有 padding：生成真的 4D mask → 我们的实现看到 mask 就退回 sdpa，结果与 sdpa 一致
    """
    ok = True
    try:
        AttentionInterface.register(NAME, sage_attention_forward)
    except Exception as e:
        print(f"[Qwen3_VL] 注册 sage attention 失败：{e}")
        ok = False

    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

        ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, sdpa_mask)
    except Exception as e:  # pragma: no cover - 取决于 transformers 版本
        print(f"[Qwen3_VL] 注册 sage 的 mask 生成器失败（带 padding 的批量会算错，单图不受影响）：{e}")

    return ok

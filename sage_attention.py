"""把 SageAttention 接进 transformers 的注意力接口。

transformers 的 `ALL_ATTENTION_FUNCTIONS` 里没有 sage，所以这里自己注册一个名字叫 "sage"
的注意力实现，然后节点就可以直接 `attn_implementation="sage"`。

设计原则：**选了什么就用什么，用不了就报错。**
--------------------------------------------
早期版本会在各种条件下静默退回 sdpa。那个设计有个致命缺陷：用户在节点上选了 `sage`，
实际跑的可能是 sdpa，**测出来的成绩是假的**（日志里也看不出来）。所以现在取消所有静默回退，
把"用不了"的情况变成显式异常，绝不偷偷换实现。

sageattn 的能力边界（本机 sageattention + sm120 实测）
----------------------------------------------------
能做的：
- GQA 原生支持（要求 num_qo_heads % num_kv_heads == 0），无需手动 repeat_kv
- `is_causal=True/False` 都支持；`is_causal` 仅在 qo_len == kv_len 时有效
- `sm_scale` 可传任意值（所以不需要再校验缩放是否等于 head_dim**-0.5）
- head_dim 无 64 倍数限制（D=64/96/100 实测均可）
- 非连续输入可以吃
不能做的（→ 抛 SageAttentionUnsupported）：
- **任意 attention_mask**：API 里根本没有 mask 参数
- dtype 非 fp16/bf16（fp32 会 AssertionError）
- 不在 CUDA 上 / dropout > 0
- 1 < q_len < kv_len（需要"带偏移的因果掩码"，sageattn 表达不了）

为什么现在连**解码**也走 sage
------------------------------
旧版本注释说"逐 token 解码有 4D mask，所以退回 sdpa"。实测这是错的：
transformers 的 `_ignore_causal_mask_sdpa()` 在"无 padding 且 (q_len == 1 或 kv_len == q_len)"
时会直接把 mask 优化成 `None`（masking_utils.py:219-262）。真实模型一次生成的调用形态实测为：

    层             q_len   kv_len   mask   is_causal   次数
    视觉塔          3844    3844    None   False        27     ← 双向
    文本预填充       972     972    None   None(=causal) 36
    文本解码           1     973+   None   None         36/token

**全程没有一个 mask**。所以解码也能走 sage：q_len==1 时用 `is_causal=False`，
单个查询看到全部 key，这正是因果解码的语义。

关于 mask 生成器的注册（必须保留）
----------------------------------
除了 `AttentionInterface`，还必须把 `ALL_MASK_ATTENTION_FUNCTIONS["sage"]` 也注册成
`sdpa_mask`。原因是 masking_utils.py:718：

    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return None      # ← 连 padding mask 一起丢掉

不注册的话，带 padding 的输入会被**静默**当成没有 padding（结果是静默算错）。
注册 sdpa_mask 后：
- 无 padding：mask 仍为 None（`_ignore_causal_mask_sdpa` 会跳过）→ sage 正常生效
- 有 padding：生成真的 4D mask → 我们的实现见到 mask 就**抛错**，而不是算错

也就是说注册它既让 sage 能生效，又给 padding 加了一道"响"的保险。

实测（同环境成绩表）
--------------------
本机 RTX 5090 / torch 2.9.1+cu130 / bf16 / 真实 Qwen3-VL 头数（Hq=32, Hkv=8, D=128）。
同一进程、同一份权重，三种 attention 交错 3 轮取中位数；greedy 强制生成 100 token
（必须用 min_new_tokens 防早停，否则贪心撞 EOS 会让"每 token 耗时"算得离谱）。

单图端到端（大图：输入 974 词元 / 961 视觉词元）：

    attn     预填充     解码/token   生成100tok总时
    eager    341.8ms    45.87ms      4928ms      ← 节点默认值
    sdpa     264.0ms    39.59ms      4223ms
    sage     170.6ms    38.39ms      4010ms      ← 三者最快

同一次生成里注意力函数自身的累计耗时（ms）：

    attn     视觉塔    文本预填充   文本解码    合计
    eager    159.5      29.7       816.7    1005.8
    sdpa      44.0      59.7       865.8     969.5
    sage      21.3       5.2       581.3     607.8

小图（输入 374 词元 / 361 视觉词元）：sage 预填充 89.8ms ≈ eager 107.1ms < sdpa 96.5ms；
注意力合计 sage 589.1 < eager 678.1 < sdpa 1040.4。

结论：**sage 在三个阶段（视觉塔 / 预填充 / 解码）全面最快，短上下文也不亏**。
所以 MIN_Q_LEN 那道门槛已删除（旧表说"1024 长度稳亏"，与实测不符）。

顺带发现：eager 拿到的是一张真的 4D mask（文本预填充和文本解码都拿），
而 sdpa/sage 走 `_ignore_causal_mask_sdpa` 直接拿到 None。这既省了建 mask 的开销，
也是 sage 能生效的前提。
"""

import torch

from transformers.modeling_utils import AttentionInterface

NAME = "sage"

_available = None  # None=还没探测过
_warned_missing = False


class SageAttentionUnsupported(RuntimeError):
    """选了 sage 但这次的调用形态 sageattn 表达不了。

    不是"退回 sdpa 的理由"，而是"必须让用户知道的错误"：
    要么改选 sdpa / eager，要么改掉触发这个形态的输入。
    """


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
                print(f"[Qwen3_VL] 未检测到 sageattention，attention=sage 会在前向时报错：{e}")
    return _available


def _unsupported(why: str, hint: str = "") -> "SageAttentionUnsupported":
    return SageAttentionUnsupported(
        f"[Qwen3_VL] attention=sage 无法处理本次调用：{why}。{hint}"
        f"（本插件不再静默退回 sdpa；要避免此错误请把节点的 attention 改成 sdpa 或 eager）"
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
    """签名与 transformers 的 sdpa_attention_forward 保持一致，返回 (output, None)。"""
    q_len = int(query.shape[2])
    kv_len = int(key.shape[2])
    n_q_heads = int(query.shape[1])
    n_kv_heads = int(key.shape[1])

    # ---------- 前置校验：不满足一律报错，不做任何回退 ----------
    if not sageattn_available():
        raise _unsupported(
            "没有安装 sageattention",
            "请 pip install sageattention（或把 attention 改成 sdpa/eager）",
        )
    if torch.jit.is_tracing() or torch.compiler.is_compiling():
        raise _unsupported(
            "当前处于 tracing / torch.compile 过程中",
            "sage 是自定义 kernel，无法被 trace；请改用 sdpa/eager，或关掉 torch.compile",
        )
    if attention_mask is not None:
        raise _unsupported(
            f"收到了 attention_mask（shape={tuple(attention_mask.shape)}），"
            f"而 sageattn 没有 mask 入参",
            "通常意味着输入里存在 padding（例如一个 batch 内序列不等长）；"
            "请对等长输入逐个跑，或改用 sdpa/eager",
        )
    if dropout:
        raise _unsupported(f"dropout={dropout}，sageattn 不支持 dropout")
    if not query.is_cuda:
        raise _unsupported(f"query 在 {query.device}，sageattn 只能在 CUDA 上跑")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise _unsupported(
            f"dtype={query.dtype}，sageattn 只接受 fp16/bf16",
            "如果是 fp32 权重，请用 dtype=float16/bfloat16 加载模型",
        )
    if n_kv_heads == 0 or n_q_heads % n_kv_heads:
        raise _unsupported(
            f"head 数不匹配（q={n_q_heads}, kv={n_kv_heads}），"
            f"sageattn 要求 num_qo_heads 能被 num_kv_heads 整除"
        )

    # ---------- 决定 causal ----------
    if is_causal is None:
        is_causal = q_len > 1 and bool(getattr(module, "is_causal", True))

    if q_len == kv_len:
        use_causal = bool(is_causal)
    elif q_len == 1:
        # 单个 query 看全部 key —— 这正是因果解码语义，所以非 causal
        use_causal = False
    else:
        raise _unsupported(
            f"q_len={q_len} < kv_len={kv_len} 且 q_len>1，"
            f"需要带位置偏移的因果掩码，sageattn 表达不了"
        )

    # ---------- 跑 sageattn ----------
    import sageattention

    out = sageattention.sageattn(
        query,
        key,
        value,
        tensor_layout="HND",
        is_causal=use_causal,
        sm_scale=scaling,       # None 时 sageattn 自己按 head_dim**-0.5 处理
        output_dtype=query.dtype,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.transpose(1, 2).contiguous(), None


def register() -> bool:
    """把 "sage" 注册进 transformers，失败不影响插件其它功能。

    要注册两张表，缺第二张会**静默算错**（见文件头"关于 mask 生成器的注册"）。
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
        print(f"[Qwen3_VL] 注册 sage 的 mask 生成器失败（带 padding 的输入会被静默算错）：{e}")

    return ok

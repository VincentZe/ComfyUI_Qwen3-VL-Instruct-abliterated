"""验证 sage_attention.py 的**严格模式**契约。

旧版测试断言的是"什么情况下退回 sdpa"。现在契约反过来了：

    选了 sage 就必须走 sage；用不了要抛错，绝不静默换成 sdpa。

判定"是否真的走了 sage"的方式：把 `sageattention.sageattn` 打桩计数。
（不再用"输出是否逐位相同"来反推路径——那个判据在新契约下不可靠，而且
sage 量化误差和 sdpa 的差异有时很小，容易误判。）

需要 GPU + 已安装 sageattention / transformers，用 ComfyUI 自己的解释器跑：

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Qwen3-VL-Instruct-abliterated/tests/test_sage_attention.py
"""
import os
import sys

import torch

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN)

import sage_attention as sa  # noqa: E402
import sageattention  # noqa: E402
from transformers.integrations.sdpa_attention import (  # noqa: E402
    sdpa_attention_forward,
)
from transformers.masking_utils import (  # noqa: E402
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
    sdpa_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

OK = []


def chk(name, cond, extra=''):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} | {name} {extra}")


# ---------------------------------------------------------------- sageattn 打桩
_calls = []
_orig_sageattn = sageattention.sageattn


def _spy(*a, **kw):
    _calls.append({
        'q_len': int(a[0].shape[2]),
        'kv_len': int(a[1].shape[2]),
        'n_q_heads': int(a[0].shape[1]),
        'n_kv_heads': int(a[1].shape[1]),
        'dtype': a[0].dtype,
        'is_causal': kw.get('is_causal'),
        'sm_scale': kw.get('sm_scale'),
    })
    return _orig_sageattn(*a, **kw)


sageattention.sageattn = _spy


def call(*args, **kw):
    """调用实现，返回 (结果, 本次调用记录列表)。"""
    _calls.clear()
    out = sa.sage_attention_forward(*args, **kw)
    return out, list(_calls)


class FakeMod(torch.nn.Module):
    def __init__(self, n_rep=4, is_causal=True):
        super().__init__()
        self.num_key_value_groups = n_rep
        self.is_causal = is_causal


H = 32
D = 128
dev = 'cuda'


def make(n_rep=4, S=974, D_=D, dtype=torch.bfloat16, device=dev):
    q = torch.randn(1, H, S, D_, dtype=dtype, device=device)
    k = torch.randn(1, max(1, H // n_rep), S, D_, dtype=dtype, device=device)
    v = torch.randn(1, max(1, H // n_rep), S, D_, dtype=dtype, device=device)
    return q, k, v


def expect_error(tag, fn, want='sage 无法处理', contains=()):
    try:
        fn()
    except sa.SageAttentionUnsupported as e:
        msg = str(e)
        ok = want in msg and all(c in msg for c in contains)
        chk(tag, ok, f'(收到 SageAttentionUnsupported: {msg[:70]}…)')
        return
    except Exception as e:  # 类型不对也要报出来
        chk(tag, False, f'(抛的是 {type(e).__name__} 而不是 SageAttentionUnsupported: {e!r})')
        return
    chk(tag, False, '(居然没抛错)')


print('=== 1. 注册与模块形状 ===')
chk('register() 成功', sa.register() is True)
chk('sageattn_available() 为真', sa.sageattn_available() is True)
chk('transformers 认可 "sage"', 'sage' in ALL_ATTENTION_FUNCTIONS.valid_keys())
chk('取到的就是我们的实现',
    ALL_ATTENTION_FUNCTIONS['sage'] is sa.sage_attention_forward)
chk('异常类型是 RuntimeError 的子类',
    issubclass(sa.SageAttentionUnsupported, RuntimeError))
chk('MIN_Q_LEN 门槛已删除（旧值 3072 是错的，会让 sage 永远不生效）',
    not hasattr(sa, 'MIN_Q_LEN'))
chk('head_dim 对齐常量已删除（实测 D=96/100/64 都能跑）',
    not hasattr(sa, 'HEAD_DIM_MULTIPLE'))

print('\n=== 1b. mask 生成器注册（决定 sage 能否生效 / padding 会不会被静默忽略）===')
chk('mask 生成器里也有 "sage"', 'sage' in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys())
chk('sage 的 mask 生成器与 sdpa 是同一个',
    ALL_MASK_ATTENTION_FUNCTIONS['sage'] is sdpa_mask)

from transformers import Qwen3VLTextConfig  # noqa: E402

_mcfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4,
                          num_key_value_heads=2, num_hidden_layers=2, vocab_size=128)
_memb = torch.zeros(1, 16, 64)
_mpos = torch.arange(16)
_mall = torch.ones(1, 16, dtype=torch.long)
_mpad = torch.ones(1, 16, dtype=torch.long)
_mpad[0, :4] = 0  # 前 4 个位置是 padding


def _mask_for(impl, am):
    _mcfg._attn_implementation = impl
    return create_causal_mask(config=_mcfg, input_embeds=_memb, attention_mask=am,
                              cache_position=_mpos, past_key_values=None,
                              position_ids=_mpos[None])


chk('sage：无 padding 的预填充不生成 mask（否则 sage 永远走不到）',
    _mask_for('sage', _mall) is None)
chk('sage：有 padding 时生成真的 4D mask（于是会抛错，而不是静默算错）',
    _mask_for('sage', _mpad) is not None)
chk('sdpa：有 padding 时行为一致',
    _mask_for('sdpa', _mpad) is not None)

print('\n=== 2. 该走 sage 的形态：必须真的调用 sageattn ===')

# 2a. 短序列（默认单图长度 ~974 词元）——旧版会把这里退回 sdpa，是主要错误
q, k, v = make(S=974)
(out, _), c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
chk('S=974（单图默认长度）→ 真的走了 sage',
    len(c) == 1, f'(sageattn 调用 {len(c)} 次)')
chk('  └ 传的是 causal=True（q_len == kv_len）',
    c and c[0]['is_causal'] is True)
chk('  └ GQA 原生传入，没有手动 repeat_kv（kv 头数仍是 8）',
    c and c[0]['n_kv_heads'] == 8 and c[0]['n_q_heads'] == 32)
chk('  └ 结果形状与 sdpa 一致', tuple(out.shape) == (1, 974, 32, 128))

# 2b. 极短序列也必须走 sage
q, k, v = make(S=64)
(out, _), c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
chk('S=64 这种极短序列也走 sage（不再有阈值门槛）', len(c) == 1)

# 2c. 解码：q_len=1, kv_len=973（真实生成里出现 36×N 次）
q, k, v = make(S=973)
q1 = q[:, :, :1, :]
(out, _), c = call(FakeMod(4), q1.clone(), k.clone(), v.clone(), None)
chk('解码 q_len=1, kv_len=973 → 走 sage（旧版这里退回 sdpa，白亏一大块）',
    len(c) == 1, f'(sageattn 调用 {len(c)} 次)')
chk('  └ 用 is_causal=False（单查询看全部 key，正是因果解码语义）',
    c and c[0]['is_causal'] is False)
chk('  └ 输出形状 (1,1,32,128)', tuple(out.shape) == (1, 1, 32, 128))

# 2d. 视觉塔：is_causal=False + q_len == kv_len
q, k, v = make(S=3844)
(out, _), c = call(FakeMod(4, is_causal=False), q.clone(), k.clone(), v.clone(), None)
chk('视觉塔 is_causal=False（双向）→ 走 sage', len(c) == 1)
chk('  └ 传的是 is_causal=False', c and c[0]['is_causal'] is False)

# 2e. 非默认 scaling 必须透传，而不是回退
q, k, v = make(S=974)
(out, _), c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None, scaling=0.01)
chk('非默认 scaling → 仍然走 sage（sm_scale 透传）', len(c) == 1)
chk('  └ sm_scale 原样传给了 sageattn', c and c[0]['sm_scale'] == 0.01)

# 2f. 未对齐的 head_dim 也要走 sage
q, k, v = make(S=256, D_=96)
(out, _), c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
chk('head_dim=96（非 64 倍数）→ 走 sage', len(c) == 1, f'(out={tuple(out.shape)})')

# 2g. MHA（无 GQA）
q, k, v = make(n_rep=1, S=974)
(out, _), c = call(FakeMod(1), q.clone(), k.clone(), v.clone(), None)
chk('MHA（num_key_value_groups=1）→ 走 sage', len(c) == 1)

# 2h. fp16
q, k, v = make(S=974, dtype=torch.float16)
(out, _), c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
chk('fp16 → 走 sage', len(c) == 1)

print('\n=== 3. 用不了的形态：必须抛 SageAttentionUnsupported ===')

q, k, v = make(S=974)
_mask4d = torch.ones(1, 1, 974, 974, dtype=torch.bool, device=dev).tril()
expect_error('带 4D mask（padding）→ 抛错，绝不静默忽略',
             lambda: call(FakeMod(4), q.clone(), k.clone(), v.clone(), _mask4d),
             contains=('attention_mask', 'sdpa'))

q32, k32, v32 = make(S=974, dtype=torch.float32)
expect_error('fp32 → 抛错（sageattn 只吃 fp16/bf16）',
             lambda: call(FakeMod(4), q32.clone(), k32.clone(), v32.clone(), None),
             contains=('fp32', 'fp16'))

qc, kc, vc = make(S=974, device='cpu')
expect_error('CPU 张量 → 抛错',
             lambda: call(FakeMod(4), qc.clone(), kc.clone(), vc.clone(), None),
             contains=('CUDA',))

q, k, v = make(S=974)
expect_error('dropout>0 → 抛错',
             lambda: call(FakeMod(4), q.clone(), k.clone(), v.clone(), None, dropout=0.1),
             contains=('dropout',))

# 1 < q_len < kv_len：需要带偏移的因果掩码，sageattn 表达不了
q, k, v = make(S=600)
q10 = q[:, :, :10, :]
expect_error('1 < q_len(10) < kv_len(600) → 抛错（偏移因果表达不了）',
             lambda: call(FakeMod(4), q10.clone(), k.clone(), v.clone(), None),
             contains=('偏移',))

# head 数不整除
q = torch.randn(1, 30, 256, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(1, 8, 256, D, dtype=torch.bfloat16, device=dev)
v = torch.randn_like(k)
expect_error('q 头数 30 不能被 kv 头数 8 整除 → 抛错',
             lambda: call(FakeMod(4), q.clone(), k.clone(), v.clone(), None),
             contains=('整除',))

# sageattn 自身失败时，错误必须冒出来（不能被吞掉再退回 sdpa）
def _boom(*a, **kw):
    raise ValueError('模拟内核崩溃')


sageattention.sageattn = _boom
try:
    call(FakeMod(4), *[t.clone() for t in make(S=974)], None)
    chk('sageattn 内核报错时必须向上抛出（不得被吞掉退回 sdpa）', False, '(居然没抛)')
except ValueError as e:
    chk('sageattn 内核报错时必须向上抛出（不得被吞掉退回 sdpa）', '模拟内核崩溃' in str(e))
except Exception as e:
    chk('sageattn 内核报错时必须向上抛出（不得被吞掉退回 sdpa）', False,
        f'(抛了 {type(e).__name__}：{e!r})')
finally:
    sageattention.sageattn = _spy

print('\n=== 4. 数值正确性（sage vs sdpa 参考）===')

# 4a. 预填充 causal：sage 与 sdpa 语义应当一致，误差是量化级别
q, k, v = make(S=974)
sage_out, c = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
ref, _ = sdpa_attention_forward(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
err = (sage_out[0].float() - ref.float()).abs().max().item()
mag = ref.float().abs().mean().item()
chk('预填充：误差在量化级别（max_err < 1.0）', err < 1.0,
    f'(max_err={err:.4f}, 参考量级={mag:.4f})')
chk('预填充：输出有限且形状一致',
    bool(torch.isfinite(sage_out[0]).all()) and sage_out[0].shape == ref.shape)
chk('预填充：确实不是逐位相同（证明真的跑了 sage 而不是 sdpa）',
    not torch.equal(sage_out[0], ref))

# 4b. 解码：q_len=1 用 is_causal=False，应与 sdpa(无 mask) 高度一致
q, k, v = make(S=973)
q1 = q[:, :, :1, :]
sage_out, c = call(FakeMod(4), q1.clone(), k.clone(), v.clone(), None)
ref, _ = sdpa_attention_forward(FakeMod(4), q1.clone(), k.clone(), v.clone(), None,
                                is_causal=False)
err = (sage_out[0].float() - ref.float()).abs().max().item()
chk('解码：与 sdpa(is_causal=False) 误差极小（< 0.05）', err < 0.05,
    f'(max_err={err:.5f})')

# 4c. 因果语义是否真的生效：causal 与非 causal 的结果必须明显不同
q, k, v = make(S=974)
kref = k.repeat_interleave(4, dim=1)
vref = v.repeat_interleave(4, dim=1)
causal_ref = torch.nn.functional.scaled_dot_product_attention(
    q, kref, vref, is_causal=True)
nocausal_ref = torch.nn.functional.scaled_dot_product_attention(
    q, kref, vref, is_causal=False)
diff = (causal_ref - nocausal_ref).abs().max().item()
sage_out, _ = call(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
err_causal = (sage_out[0].transpose(1, 2) - causal_ref).abs().max().item()
err_nocausal = (sage_out[0].transpose(1, 2) - nocausal_ref).abs().max().item()
chk('对照组：因果 vs 非因果差异远大于量化误差（说明这个检查有意义）',
    diff > 1.0, f'(差异={diff:.4f})')
chk('sage 结果更接近 causal 参考（说明 is_causal=True 真的传下去了）',
    err_causal < err_nocausal,
    f'(err_causal={err_causal:.4f} < err_nocausal={err_nocausal:.4f})')

print('\n=== 5. 视觉塔的双向语义 ===')
q, k, v = make(S=1024)
sage_out, _ = call(FakeMod(4, is_causal=False), q.clone(), k.clone(), v.clone(), None)
kref = k.repeat_interleave(4, dim=1)
vref = v.repeat_interleave(4, dim=1)
bi = torch.nn.functional.scaled_dot_product_attention(q, kref, vref, is_causal=False)
causal = torch.nn.functional.scaled_dot_product_attention(q, kref, vref, is_causal=True)
e_bi = (sage_out[0].transpose(1, 2) - bi).abs().max().item()
e_ca = (sage_out[0].transpose(1, 2) - causal).abs().max().item()
chk('is_causal=False 时结果更接近双向参考（确认没被错当 causal）',
    e_bi < e_ca, f'(err_双向={e_bi:.4f} < err_因果={e_ca:.4f})')

print('\n=== 6. batch>1 等长（拼批）也能走 sage ===')
_B, _SB = 4, 1400
qb = torch.randn(_B, H, _SB, D, dtype=torch.bfloat16, device=dev)
kb = torch.randn(_B, H // 4, _SB, D, dtype=torch.bfloat16, device=dev)
vb = torch.randn_like(kb)
outb, c = call(FakeMod(4), qb.clone(), kb.clone(), vb.clone(), None)
chk(f'batch={_B} 等长 → 走 sage', len(c) == 1)
chk('  └ 输出形状正确且有限',
    outb[0].shape == (1 * _B, _SB, H, D) and bool(torch.isfinite(outb[0]).all()),
    f'(shape={tuple(outb[0].shape)})')
refb, _ = sdpa_attention_forward(FakeMod(4), qb.clone(), kb.clone(), vb.clone(), None)
chk('  └ 与 sdpa 误差在量化级别',
    (outb[0].float() - refb.float()).abs().max().item() < 1.0)

print('\n=== 7. 带 padding 的 batch 会抛错（而不是静默忽略 padding）===')
_pad2 = torch.ones(_B, 16, dtype=torch.long)
_pad2[1, :3] = 0
_mcfg._attn_implementation = 'sage'
chk('等长 batch 不生成 4D mask',
    create_causal_mask(config=_mcfg, input_embeds=torch.zeros(_B, 16, 64),
                       cache_position=torch.arange(16),
                       attention_mask=torch.ones(_B, 16, dtype=torch.long),
                       past_key_values=None,
                       position_ids=torch.arange(16)[None].expand(_B, 16)) is None)
chk('不等长 batch（有 padding）会生成 4D mask',
    create_causal_mask(config=_mcfg, input_embeds=torch.zeros(_B, 16, 64),
                       cache_position=torch.arange(16), attention_mask=_pad2,
                       past_key_values=None,
                       position_ids=torch.arange(16)[None].expand(_B, 16)) is not None)
_m4 = torch.ones(_B, 1, _SB, _SB, dtype=torch.bool, device=dev).tril()
expect_error('batch>1 + 4D mask（真实 padding）→ 抛错',
             lambda: call(FakeMod(4), qb.clone(), kb.clone(), vb.clone(), _m4),
             contains=('attention_mask',))

# 还原打桩，避免影响其它测试
sageattention.sageattn = _orig_sageattn

print(f"\n结果: {sum(OK)}/{len(OK)} 通过")
sys.exit(0 if all(OK) else 1)

"""验证 sage_attention.py：注册是否成功、哪些情况走 sage、哪些自动退回 sdpa。

需要 GPU + 已安装 sageattention / transformers，用 ComfyUI 自己的解释器跑：

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Qwen3-VL-Instruct-abliterated/tests/test_sage_attention.py

判定方式：退回 sdpa 时输出与 transformers 的 sdpa_attention_forward 逐位相同（误差 0），
走 sage 时因为 int8/fp8 量化会有一个小误差（约 0.2），靠这个区分两条路径。

注意：sage 有 MIN_Q_LEN 门槛（默认 3072，见 sage_attention.py 文件头的实测表），
所以"期望走 sage"的用例必须用长序列（S_LONG），短序列一律应该退回 sdpa。
"""
import importlib
import os
import sys

import torch

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN)

import sage_attention as sa  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
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


print('=== 1. 注册 ===')
chk('register() 成功', sa.register() is True)
chk('sageattn_available() 为真', sa.sageattn_available() is True)
keys = ALL_ATTENTION_FUNCTIONS.valid_keys()
chk('transformers 认可 "sage"', 'sage' in keys)
fn = ALL_ATTENTION_FUNCTIONS['sage']
chk('取到的就是我们的实现', fn is sa.sage_attention_forward)
chk('MIN_Q_LEN 默认值达到"每轮都赚钱"的长度（>=3072）', sa.MIN_Q_LEN >= 3072, f'(={sa.MIN_Q_LEN})')


print('=== 1b. mask 生成器注册（决定 sage 能不能生效 / padding 会不会被丢）===')
chk('mask 生成器里也有 "sage"', 'sage' in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys())
chk('sage 的 mask 生成器与 sdpa 是同一个', ALL_MASK_ATTENTION_FUNCTIONS['sage'] is sdpa_mask)

from transformers import Qwen3VLTextConfig  # noqa: E402

_mcfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                          num_hidden_layers=2, vocab_size=128)
_memb = torch.zeros(1, 16, 64)
_mpos = torch.arange(16)
_mall = torch.ones(1, 16, dtype=torch.long)
_mpad = torch.ones(1, 16, dtype=torch.long)
_mpad[0, :4] = 0  # 真实 padding：前 4 个位置是补齐的


def _mask_for(impl, am):
    _mcfg._attn_implementation = impl
    return create_causal_mask(config=_mcfg, input_embeds=_memb, attention_mask=am,
                              cache_position=_mpos, past_key_values=None,
                              position_ids=_mpos[None])


chk('sage：无 padding 的预填充不生成 mask（否则 sage 永远走不到）',
    _mask_for('sage', _mall) is None)
chk('sage：有 padding 时生成 4D mask（不再静默丢弃 padding）',
    _mask_for('sage', _mpad) is not None)
chk('sdpa：有 padding 时行为一致（也生成 mask）',
    _mask_for('sdpa', _mpad) is not None)


class FakeMod(torch.nn.Module):
    def __init__(self, n_rep=4, is_causal=True):
        super().__init__()
        self.num_key_value_groups = n_rep
        self.is_causal = is_causal


H = 32
dev = 'cuda'
S_LONG = 4096      # 超过 MIN_Q_LEN，应该走 sage
S_DEFAULT_IMG = 1400  # 默认 max_pixels=1280*28*28 ≈ 1280 视觉词元，应退回 sdpa


def make(n_rep=4, S=S_LONG, D=128, dtype=torch.float16):
    q = torch.randn(1, H, S, D, dtype=dtype, device=dev)
    k = torch.randn(1, max(1, H // n_rep), S, D, dtype=dtype, device=dev)
    v = torch.randn(1, max(1, H // n_rep), S, D, dtype=dtype, device=dev)
    return q, k, v


def run_case(tag, *, expect_sage, n_rep=4, S=S_LONG, D=128, dtype=torch.float16,
             mask=None, is_causal=None, scale=None, mod=None):
    q, k, v = make(n_rep, S, D, dtype)
    m = mod or FakeMod(n_rep)
    out, _ = fn(m, q.clone(), k.clone(), v.clone(), mask, scaling=scale, is_causal=is_causal)
    ref, _ = sdpa_attention_forward(m, q.clone(), k.clone(), v.clone(), mask,
                                    scaling=scale, is_causal=is_causal)
    identical = torch.equal(out, ref)
    err = (out.float() - ref.float()).abs().max().item()
    if expect_sage:
        good = (not identical) and err < 0.6 and bool(torch.isfinite(out).all())
    else:
        good = identical
    chk(tag, good, f"(走sage={not identical}, max_err={err:.4f})")
    return out


print('\n=== 2. 路径选择 ===')
run_case(f'超长序列 S={S_LONG} → 走 sage（GQA 4:1, fp16, D=128）', expect_sage=True)
run_case('MHA（num_key_value_groups=1）→ 走 sage', expect_sage=True, n_rep=1)
run_case('bf16 → 走 sage', expect_sage=True, dtype=torch.bfloat16)
run_case(f'默认单图长度 S={S_DEFAULT_IMG} → 退回 sdpa（sage 在此长度更慢）',
         expect_sage=False, S=S_DEFAULT_IMG)
run_case('S=8 太短 → 退回 sdpa', expect_sage=False, S=8)
run_case(f'恰好等于阈值 S={sa.MIN_Q_LEN} → 走 sage', expect_sage=True, S=sa.MIN_Q_LEN)
run_case(f'比阈值少 1 S={sa.MIN_Q_LEN - 1} → 退回 sdpa', expect_sage=False, S=sa.MIN_Q_LEN - 1)
run_case('带 4D mask（解码步）→ 退回 sdpa', expect_sage=False,
         mask=torch.ones(1, 1, S_LONG, S_LONG, dtype=torch.bool, device=dev).tril())
run_case('fp32 → 退回 sdpa', expect_sage=False, dtype=torch.float32)
run_case('is_causal=False（视觉塔）→ 退回 sdpa', expect_sage=False, is_causal=False)
run_case('head_dim=96 未对齐 64 → 退回 sdpa', expect_sage=False, D=96)
run_case('非默认 scaling → 退回 sdpa', expect_sage=False, scale=0.01)

print('\n=== 3. q_len != k_len 的显式构造 ===')
q = torch.randn(1, H, S_LONG, 128, dtype=torch.float16, device=dev)
k = torch.randn(1, H // 4, S_LONG * 2, 128, dtype=torch.float16, device=dev)
v = torch.randn(1, H // 4, S_LONG * 2, 128, dtype=torch.float16, device=dev)
m = FakeMod(4)
out, _ = fn(m, q.clone(), k.clone(), v.clone(), None)
ref, _ = sdpa_attention_forward(m, q.clone(), k.clone(), v.clone(), None)
chk('q_len != k_len → 退回 sdpa（避免错用 causal）', torch.equal(out, ref))

print('\n=== 4. 正确性（sage 路径 vs sdpa 参考） ===')
q, k, v = make(4, S_LONG, 128)
out, _ = fn(FakeMod(4), q.clone(), k.clone(), v.clone(), None)
kref = k.repeat_interleave(4, dim=1)
vref = v.repeat_interleave(4, dim=1)
ref = torch.nn.functional.scaled_dot_product_attention(q, kref, vref, is_causal=True)
err = (out.transpose(1, 2) - ref).abs().max().item()
chk('causal 语义正确（误差是量化级别）', err < 0.6, f'max_err={err:.4f}')

nocausal = torch.nn.functional.scaled_dot_product_attention(q, kref, vref, is_causal=False)
diff = (ref - nocausal).abs().max().item()
chk('对照组：因果/非因果差异远大于量化误差', diff > 1.0, f'因果 vs 非因果差={diff:.4f}')

print('\n=== 5. 环境变量可覆盖阈值 ===')
_old = os.environ.get('QWEN3_VL_SAGE_MIN_Q_LEN')
try:
    os.environ['QWEN3_VL_SAGE_MIN_Q_LEN'] = '0'
    sa = importlib.reload(sa)
    chk('QWEN3_VL_SAGE_MIN_Q_LEN=0 生效', sa.MIN_Q_LEN == 0, f'(={sa.MIN_Q_LEN})')

    os.environ['QWEN3_VL_SAGE_MIN_Q_LEN'] = '8192'
    sa = importlib.reload(sa)
    chk('QWEN3_VL_SAGE_MIN_Q_LEN=8192 生效', sa.MIN_Q_LEN == 8192, f'(={sa.MIN_Q_LEN})')
finally:
    if _old is None:
        os.environ.pop('QWEN3_VL_SAGE_MIN_Q_LEN', None)
    else:
        os.environ['QWEN3_VL_SAGE_MIN_Q_LEN'] = _old
    sa = importlib.reload(sa)
    chk('恢复默认值', sa.MIN_Q_LEN == 3072, f'(=3072 期望值, 实际={sa.MIN_Q_LEN})')

print('\n=== 6. batch>1：等长可走 sage，带 padding 会被挡回 sdpa ===')
# 实测（RTX 5090，fp16，D=128，causal，MHA）：sage 的盈亏点会随 batch 变小。
#   S=1280: B=1 →0.38x(亏)  B=4 →1.04x(平)  B=8 →1.20x(赚)
#   S=2048: B=1 →1.00x(平)  B=4 →1.71x     B=8 →2.05x
# 所以"拼批"确实让 sage 变得划算；但前提是同一批内序列**等长**，否则 padding 会产生 4D mask。
_B, _SB = 4, 1400
_emb2 = torch.zeros(_B, 16, 64)
_pos2 = torch.arange(16)
_pos2b = _pos2[None].expand(_B, 16)
_mcfg._attn_implementation = 'sage'
chk('等长 batch（全 1 mask）不生成 4D mask → sage 拼批可跑',
    create_causal_mask(config=_mcfg, input_embeds=_emb2, cache_position=_pos2,
                       attention_mask=torch.ones(_B, 16, dtype=torch.long),
                       past_key_values=None, position_ids=_pos2b) is None)
_pad2 = torch.ones(_B, 16, dtype=torch.long)
_pad2[1, :3] = 0
chk('带 padding 的 batch 会生成 4D mask → sage 退回 sdpa（不会被静默忽略）',
    create_causal_mask(config=_mcfg, input_embeds=_emb2, cache_position=_pos2,
                       attention_mask=_pad2,
                       past_key_values=None, position_ids=_pos2b) is not None)

_qb = torch.randn(_B, H, _SB, 128, dtype=torch.float16, device=dev)
_kb = torch.randn(_B, H // 4, _SB, 128, dtype=torch.float16, device=dev)
_vb = torch.randn_like(_kb)
_bk = sa.MIN_Q_LEN
sa.MIN_Q_LEN = 0  # 强制让 sage 上线，专测 batch 路径本身
try:
    _out, _ = fn(FakeMod(4), _qb.clone(), _kb.clone(), _vb.clone(), None)
    _ref, _ = sdpa_attention_forward(FakeMod(4), _qb.clone(), _kb.clone(), _vb.clone(), None)
    _ident = torch.equal(_out, _ref)
    _err = (_out.float() - _ref.float()).abs().max().item()
    chk(f'batch={_B} 等长 → 真的走 sage 且结果形状/数值正常',
        (not _ident) and _err < 0.6 and bool(torch.isfinite(_out).all()) and _out.shape == _ref.shape,
        f'(走sage={not _ident}, max_err={_err:.4f}, shape={tuple(_out.shape)})')
finally:
    sa.MIN_Q_LEN = _bk

_mask4d = torch.ones(_B, 1, _SB, _SB, dtype=torch.bool, device=dev).tril()
_outp, _ = fn(FakeMod(4), _qb.clone(), _kb.clone(), _vb.clone(), _mask4d)
_refp, _ = sdpa_attention_forward(FakeMod(4), _qb.clone(), _kb.clone(), _vb.clone(), _mask4d)
chk('batch>1 + 4D mask → 逐位等于 sdpa（padding 位不会被算进去）', torch.equal(_outp, _refp))

print(f"\n结果: {sum(OK)}/{len(OK)} 通过")
sys.exit(0 if all(OK) else 1)

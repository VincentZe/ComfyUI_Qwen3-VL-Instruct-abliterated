"""验证 vision_patch_fix：用**真实 processor 输出**验证等价性 + 实测提速。

背景：Qwen3-VL 视觉塔的 patch_embed 是 kernel==stride 的无重叠 Conv3d，
在 bf16 下本机会被 cuDNN 选到病态算法，单次 25 秒（整个视觉塔 32.8s 里占 32.6s）。
本测试证明改写版数值等价且快几个数量级。
"""
import os
import sys
import time

import torch
import torch.nn as nn

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = r"T:\NovelAI\ComfyUI\models\prompt_generator\Huihui-Qwen3-VL-8B-Instruct-abliterated"
# 测试图跟测试放一起（不要放 temp/，ComfyUI 会定期清空）
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bench_img_0.png")
sys.path.insert(0, PLUGIN)
import vision_patch_fix as vpf  # noqa: E402

from transformers import AutoProcessor  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

dev = "cuda"
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} | {name} {extra}", flush=True)


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) / iters * 1000)
    ts.sort()
    return ts[len(ts) // 2]


print("加载 processor…", flush=True)
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
msgs = [[{"role": "user", "content": [{"type": "image", "image": f"file://{IMG}"},
                                      {"type": "text", "text": "描述"}]}]]
texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
imgs, vids = process_vision_info(msgs)
inp = proc(text=texts, images=imgs, videos=vids, return_tensors="pt")
pv = inp["pixel_values"].to(dev)                 # 模型真正喂给 patch_embed 的张量
print("pixel_values:", tuple(pv.shape), pv.dtype, flush=True)

conv = nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16)).to(dev, torch.float32)
fixed = vpf.FlattenedPatchEmbed(conv).to(dev, torch.float32)

print("\n=== 1. 数值等价（关掉 TF32，让 Conv3d 走精确 fp32）===", flush=True)
_old_tf32 = torch.backends.cudnn.allow_tf32
torch.backends.cudnn.allow_tf32 = False
with torch.no_grad():
    ref = conv(pv.view(-1, 3, 2, 16, 16)).reshape(-1, 1152)
    out = fixed(pv)
    ref64 = ref.double()
torch.backends.cudnn.allow_tf32 = _old_tf32
err = (ref - out).abs().max().item()
chk("fp32 改写后 == Conv3d", err < 1e-3, f"(max_abs_err={err:.2e}, 相对={err/ref.abs().max().item():.1e})")
chk("输出形状一致", tuple(out.shape) == tuple(ref.shape), f"{tuple(out.shape)}")

print("\n=== 2. bf16（模型默认精度）等价性与速度 ===", flush=True)
cb = nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
cb.load_state_dict(conv.state_dict())
cb = cb.to(dev, torch.bfloat16)
fb = vpf.FlattenedPatchEmbed(cb).to(dev, torch.bfloat16)
pvb = pv.to(torch.bfloat16)
with torch.no_grad():
    o_lin = fb(pvb).float()
e_lin = (o_lin.double() - ref64).abs().max().item()
chk("bf16 改写版误差在舍入级", e_lin < 0.5, f"(max_abs_err={e_lin:.4f})")

t_conv_fp32 = bench(lambda: conv(pv.view(-1, 3, 2, 16, 16)))
t_lin = bench(lambda: fb(pvb))
print(f"  Conv3d fp32        : {t_conv_fp32:9.2f}ms", flush=True)
print(f"  展平 Linear bf16   : {t_lin:9.3f}ms", flush=True)
chk("改写版比 fp32 卷积更快", t_lin < t_conv_fp32, f"({t_conv_fp32:.2f}ms → {t_lin:.3f}ms)")

with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cb(pv.view(-1, 3, 2, 16, 16).to(torch.bfloat16))
    torch.cuda.synchronize()
    t_conv_bf16 = (time.perf_counter() - t0) * 1000
print(f"  Conv3d bf16（单次）: {t_conv_bf16:9.1f}ms   ← 病态路径，就是它拖慢一切", flush=True)
chk("bf16 Conv3d 确实病态慢（>1s）", t_conv_bf16 > 1000, f"({t_conv_bf16:.0f}ms)")
chk("改写后提速 >1000 倍", t_conv_bf16 / t_lin > 1000, f"({t_conv_bf16:.0f}ms → {t_lin:.3f}ms, {t_conv_bf16/t_lin:.0f}x)")

print("\n=== 3. 输入形态 ===", flush=True)
# 模型内部会先 view 成 (N, C, T, ps, ps) 再送进来，这是展平 patch 的视图，必须支持
pv5 = pv.view(-1, 3, 2, 16, 16)
with torch.no_grad():
    o5 = fixed(pv5)
chk("5D（展平 patch 的视图）可用且与 2D 一致", torch.equal(o5, out), f"{tuple(pv5.shape)}")
try:
    fixed(torch.randn(1, 3, 2, 768, 1024, dtype=torch.float32, device=dev))
    chk("原始图像张量应被拒绝（避免静默算错）", False)
except ValueError:
    chk("原始图像张量应被拒绝（避免静默算错）", True)


class _Dummy(nn.Module):
    class _M(nn.Module):
        class _V(nn.Module):
            class _PE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4))

            def __init__(self):
                super().__init__()
                self.patch_embed = self._PE()

        def __init__(self):
            super().__init__()
            self.visual = self._V()

    def __init__(self):
        super().__init__()
        self.model = self._M()


print("\n=== 4. apply() 挂载 ===", flush=True)
dummy = _Dummy()
chk("apply 替换 1 个模块", vpf.apply(dummy, verbose=False) == 1)
chk("替换后类型正确", isinstance(dummy.model.visual.patch_embed.proj, vpf.FlattenedPatchEmbed))
chk("apply 幂等", vpf.apply(dummy, verbose=False) == 0)
try:
    vpf.FlattenedPatchEmbed(nn.Conv3d(3, 8, (3, 3, 3), stride=(1, 1, 1)))
    chk("重叠卷积应被拒绝", False)
except ValueError:
    chk("重叠卷积应被拒绝", True)

print("\n=== 5. 安全守卫 ===", flush=True)
# 5a. 环境变量整体关闭
os.environ["QWEN3_VL_NO_PATCH_FIX"] = "1"
try:
    d = _Dummy()
    chk("QWEN3_VL_NO_PATCH_FIX=1 → apply 不动手", vpf.apply(d, verbose=False) == 0)
    chk("  ...且模块仍是 Conv3d", isinstance(d.model.visual.patch_embed.proj, nn.Conv3d))
finally:
    os.environ.pop("QWEN3_VL_NO_PATCH_FIX", None)
chk("清掉环境变量后 apply 恢复工作", vpf.apply(_Dummy(), verbose=False) == 1)

# 5b. 多设备（accelerate 卸载）→ 跳过，避免破坏 offload hook
d2 = _Dummy()
d2.hf_device_map = {"model.visual": 0, "model.language_model": 1}
chk("多设备 hf_device_map → 跳过改写", vpf.apply(d2, verbose=False) == 0)
chk("  ...模块未被替换", isinstance(d2.model.visual.patch_embed.proj, nn.Conv3d))

# 5c. 单设备 device_map 视为安全，照常改写
d3 = _Dummy()
d3.hf_device_map = {"model.visual": 0, "model.language_model": 0}
chk("单设备 device_map → 照常改写", vpf.apply(d3, verbose=False) == 1)

# 5d. 非 patch_embed 的 Conv3d 不被动（只认 patch_embed.proj）
d4 = _Dummy()
d4.other_conv = nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4))
chk("只替换 patch_embed.proj，放过其它 Conv3d",
    vpf.apply(d4, verbose=False) == 1 and isinstance(d4.other_conv, nn.Conv3d))

# 5e. 权重仍在 meta 设备（磁盘/CPU 卸载未加载）→ 跳过，避免造出 meta 权重在前向报错
d5 = _Dummy()
with torch.device("meta"):
    d5.model.visual.patch_embed.proj = nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4))
chk("meta 权重 → 跳过改写", vpf.apply(d5, verbose=False) == 0)
chk("  ...模块未被替换", not isinstance(d5.model.visual.patch_embed.proj, vpf.FlattenedPatchEmbed))

print(f"\n结果: {sum(OK)}/{len(OK)} 通过", flush=True)
sys.exit(0 if all(OK) else 1)

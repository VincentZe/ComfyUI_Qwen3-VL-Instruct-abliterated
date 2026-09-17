"""把 Qwen3-VL 视觉塔的 patch_embed 从 Conv3d 换成等价的展平 Linear。

为什么需要：patch_embed 是 kernel == stride 的无重叠 Conv3d，数学上就是一次
"把 patch 展平后做矩阵乘"。但本机（RTX 5090 / torch 2.9.1+cu130 / cuDNN 91200）
在 **bf16** 下会选到一个病态卷积算法，单次调用要 **25 秒**；整个视觉塔 32.8 秒里
32.6 秒都耗在这里（27 层 transformer 加起来才 40ms）。

实测同一形状：
    Conv3d bf16        : 25 700 ms
    Conv3d fp32        :      1.4 ms
    展平后 Linear bf16 :      0.09 ms

所以这里把 Conv3d 就地替换成等价实现，数值上只差 bf16 舍入（数学完全等价）。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenedPatchEmbed(nn.Module):
    """与 kernel_size == stride 的 Conv3d 数学等价，但走 GEMM / tensor core。

    只接受 Qwen3-VL 实际使用的输入形态：`(num_patches, C*T*ps*ps)`——processor 输出的
    `pixel_values` 本身就是这个形状，模型里是 `self.patch_embed(hidden_states)`
    （等价于 `hidden_states.view(-1, C, T, ps, ps)` 再卷积）。

    因为 kernel == stride、无重叠、无 padding，每个 patch 的输出就是该 patch 展平向量
    与权重矩阵的乘积，所以 `reshape(-1, C*T*ps*ps) @ W.T` 完全等价。
    （不要传 5D 图像张量：那时 patch 在内存里不连续，展平顺序与卷积不一致，
    会静默算错，所以这里显式报错。）
    """

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        if tuple(conv.kernel_size) != tuple(conv.stride):
            raise ValueError("只支持无重叠卷积（kernel_size == stride）")
        if conv.groups != 1 or conv.dilation != (1, 1, 1):
            raise ValueError("只支持 groups=1 / dilation=1 的卷积")

        self.in_channels = conv.in_channels
        self.temporal_patch_size = conv.kernel_size[0]
        self.patch_size = conv.kernel_size[1]

        # (out, in, kt, kh, kw) -> (out, in*kt*kh*kw)，与输入展平顺序一致
        self.weight = nn.Parameter(conv.weight.detach().reshape(conv.out_channels, -1).clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.in_features = self.weight.shape[1]
        self.out_features = self.weight.shape[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """接受两种等价输入：

        1. `(num_patches, C*T*ps*ps)` —— processor 的 `pixel_values` 原样。
        2. `(num_patches, C, T, ps, ps)` —— 模型内部 `hidden_states.view(-1, C, T, ps, ps)`
           的结果（`modeling_qwen3_vl.py:72-75`）。它是展平 patch 的**视图**，
           所以 patch 数据在内存里连续，直接展平即可，与卷积结果一致。

        真正的原始图像张量（如 `(1, 3, 2, 768, 1024)`，patch 在内存里不连续）
        会和卷积语义不一致，这里直接拒绝，避免静默算错。
        """
        if hidden_states.dim() == 5:
            b, c, t, h, w = hidden_states.shape
            if (h, w) != (self.patch_size, self.patch_size) or t != self.temporal_patch_size:
                raise ValueError(
                    f"收到 5D 形状 {tuple(hidden_states.shape)}，它不是展平 patch 的视图"
                    f"（应为 (num_patches, {self.in_channels}, {self.temporal_patch_size}, "
                    f"{self.patch_size}, {self.patch_size})）。原始图像张量的 patch 在内存里"
                    f"不连续，需要先按 processor 的方式切分展平。"
                )
        elif hidden_states.dim() != 2:
            raise ValueError(f"不支持的输入形状 {tuple(hidden_states.shape)}")

        x = hidden_states.reshape(-1, self.in_features)
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias)


def apply(model, verbose: bool = True) -> int:
    """就地替换所有 patch_embed 里的无重叠 Conv3d。返回替换个数。

    设置环境变量 `QWEN3_VL_NO_PATCH_FIX=1` 可整体关闭（用于排查）。
    """
    if os.environ.get("QWEN3_VL_NO_PATCH_FIX", "").strip() in ("1", "true", "yes"):
        if verbose:
            print("[Qwen3_VL] 已按环境变量 QWEN3_VL_NO_PATCH_FIX 跳过 patch_embed 改写")
        return 0

    # 注意：不要按"device_map 有多个设备"一刀切跳过。部分卸载时视觉塔可能
    # 仍完整驻留在 GPU 上（hf_device_map 里 'model.visual': 0），那正是
    # bf16 Conv3d 病态慢的重灾区。安全的判定是逐模块看：
    #   - 权重还在 meta 设备（卸载尚未加载）→ 不能碰；
    #   - 模块挂着 accelerate 的 _hf_hook（offload 由 hook 在前向时搬权重）
    #     → 换掉模块会丢 hook，不能碰。
    # 其余情况（真实驻留在某块设备上、无 hook）替换成等价 Linear 不会影响
    # 其它层的 offload 路径。

    replaced = 0
    skipped_meta = 0
    skipped_hook = 0
    for name, module in list(model.named_modules()):
        if not name.endswith("patch_embed.proj"):
            continue
        if not isinstance(module, nn.Conv3d):
            continue
        if tuple(module.kernel_size) != tuple(module.stride):
            continue
        # 权重还在 meta 设备（磁盘/CPU 卸载尚未加载）→ 不能碰，否则新模块也是 meta，
        # 前向会直接报错。跳过：慢，但不会崩。
        if module.weight.is_meta:
            skipped_meta += 1
            continue
        # accelerate offload hook 挂在模块上（前向时由 hook 搬权重）→ 换掉模块
        # 会把 hook 一起丢掉，新模块会滞留在 cpu 上算。跳过：慢，但不会错。
        if hasattr(module, "_hf_hook"):
            skipped_hook += 1
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        new = FlattenedPatchEmbed(module).to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, "proj", new)
        replaced += 1
        if verbose:
            print(f"[Qwen3_VL] patch_embed 已改写为展平 Linear：{name} "
                  f"({module.out_channels} x {new.in_features}, {module.weight.dtype})")

    if verbose and skipped_meta:
        print(f"[Qwen3_VL] {skipped_meta} 个 patch_embed 权重仍在 meta 设备（未加载），跳过改写")
    if verbose and skipped_hook:
        print(f"[Qwen3_VL] {skipped_hook} 个 patch_embed 挂着 accelerate offload hook，跳过改写")
    if verbose and not replaced and not skipped_meta:
        print("[Qwen3_VL] 未找到可改写的 patch_embed Conv3d（版本可能已变，跳过）")
    return replaced

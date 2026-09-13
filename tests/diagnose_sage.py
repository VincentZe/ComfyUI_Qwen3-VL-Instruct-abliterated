"""sageattention 环境诊断：装了哪个包、编译了哪些架构、5090 上实际走哪条内核、值不值得用。

不需要 GPU 也能跑前半部分；后半部分（实跑 + 测速）需要 GPU。

    .venv/Scripts/python.exe custom_nodes/ComfyUI_Qwen3-VL-Instruct-abliterated/tests/diagnose_sage.py

排查"某个插件报 sage 相关错误"时先跑这个。
"""
import os
import re
import struct
import sys
import time
import traceback

SEP = "=" * 70


# --------------------------------------------------------------------------
# 1. 环境
# --------------------------------------------------------------------------
def dump_env():
    print(SEP)
    print("1. 环境")
    print(SEP)
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch

        print(f"torch       : {torch.__version__}  (CUDA {torch.version.cuda})")
        print(f"torch archs : {torch.cuda.get_arch_list()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU         : {p.name}  cc={p.major}.{p.minor}  "
                  f"{p.multi_processor_count} SMs  {p.total_memory / 1024**3:.1f} GB")
    except Exception:
        print("torch 探测失败:")
        traceback.print_exc()

    try:
        import triton

        print(f"triton      : {triton.__version__}")
    except Exception as e:
        print(f"triton      : 不可用 ({e})")

    try:
        import sageattention

        print(f"sageattention: {sageattention.__file__}")
    except Exception as e:
        print(f"sageattention: 导入失败 ({e})")

    import shutil

    for t in ("nvcc", "cl", "ninja"):
        print(f"{t:<12}: {shutil.which(t) or '未找到'}")


# --------------------------------------------------------------------------
# 2. 二进制里编了哪些架构
# --------------------------------------------------------------------------
def scan_archs():
    print()
    print(SEP)
    print("2. sageattention 的 .pyd 里内嵌了哪些 GPU 架构")
    print(SEP)
    try:
        import sageattention

        d = os.path.dirname(sageattention.__file__)
    except Exception as e:
        print(f"跳过：{e}")
        return None

    target_re = re.compile(rb"\.target\s+(sm_\d+[a-z]?|compute_\d+[a-z]?)")
    found_any_sm120 = False

    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pyd"):
            continue
        data = open(os.path.join(d, fn), "rb").read()

        ptx = sorted({m.decode() for m in target_re.findall(data)})

        archs = {}
        off = 0
        while True:
            off = data.find(b"\x7fELF", off)
            if off < 0:
                break
            off += 4
            try:
                if struct.unpack_from("<H", data, off + 14)[0] == 190:  # EM_CUDA
                    fl = struct.unpack_from("<I", data, off + 44)[0]
                    sm = fl & 0xFFFF
                    key = f"sm_{(sm >> 8) & 0xFF}{sm & 0xFF}"
                    archs[key] = archs.get(key, 0) + 1
            except Exception:
                pass

        names = sorted(archs)
        if any(n.startswith("sm_120") for n in names):
            found_any_sm120 = True
        cubin_str = ", ".join(f"{k}x{v}" for k, v in sorted(archs.items())) or "无"
        print(f"\n{fn}  ({len(data) / 1024 / 1024:.1f} MB)")
        print(f"  cubin : {cubin_str}")
        print(f"  PTX   : {', '.join(ptx) if ptx else '（无 PTX，纯 cubin）'}")

    print()
    print("结论: " + ("发现 sm_120（Blackwell/5090）原生 kernel，无需重新编译"
                     if found_any_sm120 else
                     "没有 sm_120 kernel → 5090 上会报 'no kernel image'，需要重新编译或换 wheel"))
    return found_any_sm120


# --------------------------------------------------------------------------
# 3. 实跑：分派到哪条实现
# --------------------------------------------------------------------------
def run_dispatch():
    print()
    print(SEP)
    print("3. 实跑：sageattn() 在真实 GPU 上分派到哪条实现")
    print(SEP)
    try:
        import torch
        from sageattention import core
    except Exception as e:
        print(f"跳过：{e}")
        return

    if not torch.cuda.is_available():
        print("跳过：没有可用 GPU")
        return

    print(f"arch 判定      : {core.get_cuda_arch_versions()}")
    print(f"kernel 可用性  : SM80={core.SM80_ENABLED} SM89={core.SM89_ENABLED} "
          f"SM90={core.SM90_ENABLED}")

    called = []
    cands = ("sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton",
             "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_sm90")
    originals = {n: getattr(core, n) for n in cands}
    for n, o in originals.items():
        def mk(name, orig):
            def w(*a, **kw):
                called.append(name)
                return orig(*a, **kw)
            return w
        setattr(core, n, mk(n, o))

    try:
        q = torch.randn(1, 8, 4096, 128, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = core.sageattn(q, k, v, tensor_layout="HND", is_causal=True)
        torch.cuda.synchronize()
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"分派到         : {called or '（没走任何 sage 实现，可能静默退回）'}")
        print(f"执行结果       : PASS  与 sdpa 最大误差 {err:.4f}"
              f"（非零=真的跑了量化内核）")
    except Exception:
        print("执行结果       : FAIL")
        traceback.print_exc()
    finally:
        for n, o in originals.items():
            setattr(core, n, o)


# --------------------------------------------------------------------------
# 4. 值不值得用：sage vs sdpa 交叉点
# --------------------------------------------------------------------------
def bench_speed():
    print()
    print(SEP)
    print("4. sage vs sdpa 速度对比（决定 MIN_Q_LEN 该设多少）")
    print(SEP)
    try:
        import torch
        from sageattention import core
    except Exception as e:
        print(f"跳过：{e}")
        return

    if not torch.cuda.is_available():
        print("跳过：没有可用 GPU")
        return

    def bench(fn, iters=200, warmup=30):
        """取多轮中位数。注意：取样太少会得出错误的交叉点（曾出现 2048 与 1024 互换的假象）。"""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        rounds = 5
        per = max(1, iters // rounds)
        samples = []
        for _ in range(rounds):
            t = time.perf_counter()
            for _ in range(per):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t) / per * 1000)
        samples.sort()
        return samples[len(samples) // 2]

    print(f"{'S':>6} {'sage(ms)':>10} {'sdpa(ms)':>10} {'加速比':>8}  结论")
    print("-" * 52)
    for S in (512, 1024, 1536, 2048, 2560, 3072, 4096):
        q = torch.randn(1, 8, S, 128, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ts = bench(lambda: core.sageattn(q, k, v, tensor_layout="HND", is_causal=True))
        td = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True))
        r = td / ts
        print(f"{S:>6} {ts:>10.3f} {td:>10.3f} {r:>7.2f}x  "
              f"{'sage 更快' if r > 1.05 else ('差不多' if r > 0.95 else 'sage 更慢')}")

    print()
    print("判读：本机后台常驻 LDPlayer / 浏览器 / NVIDIA Overlay，sage 单次仅 0.2~0.5ms，")
    print("      实测耗时呈双峰波动（同一 S 不同进程可能差 2 倍），sdpa 侧则很稳定。")
    print("      所以阈值取'第一档每轮都赚钱'的长度 = 3072，不要按平均交叉点取 2048~2560。")
    print("      默认 max_pixels=1280*28*28 → 约 1280 视觉词元，落在 sage 亏本区间：")
    print("      单图 VQA 用 sage 无收益，视频/多图/长提示词（>3k 词元）才有意义。")


if __name__ == "__main__":
    dump_env()
    scan_archs()
    run_dispatch()
    bench_speed()

import os
import json
import datetime
import threading
import weakref
import torch
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import comfy.model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path

try:
    from server import PromptServer
    from aiohttp import web
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

# 把 sage 注册进 transformers 的注意力接口；注意节点可能以包内模块或顶层模块两种方式加载
try:
    from .sage_attention import register as _register_sage_attention
    from .sage_attention import sageattn_available as _sageattn_available
except ImportError:  # 直接以顶层模块导入时（例如离线测试）
    from sage_attention import register as _register_sage_attention
    from sage_attention import sageattn_available as _sageattn_available

_SAGE_REGISTERED = _register_sage_attention()

# 把视觉塔的 patch_embed 从 Conv3d 换成等价的展平 Linear（见 vision_patch_fix.py）
# 本机 bf16 Conv3d 单次要 ~25 秒（占整个视觉塔 99%），改写后 ~0.08ms
try:
    from .vision_patch_fix import apply as _apply_vision_patch_fix
except ImportError:  # 直接以顶层模块导入时（例如离线测试）
    try:
        from vision_patch_fix import apply as _apply_vision_patch_fix
    except ImportError:
        _apply_vision_patch_fix = None


CACHE_SUFFIX = ".json"  # foo.jpeg -> foo.jpeg.json
CACHE_ENTRY_PREFIX = ""  # 序号字符串直接当 id
ID_DATE_FMT = "%Y.%m.%d"
SUMMARY_MAX = 40  # 下拉标签用的提示词摘要长度

MODEL_CHOICES = [
    "Qwen3-VL-4B-Instruct-FP8",
    "Qwen3-VL-4B-Thinking-FP8",
    "Qwen3-VL-8B-Instruct-FP8",
    "Qwen3-VL-8B-Thinking-FP8",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "Huihui-Qwen3-VL-8B-Instruct-abliterated",
]
# 注意：上面这个列表现在只是**兜底**。上游（f1061fe）把模型下拉改成了动态扫描
# models/prompt_generator 下名字含 "Qwen3-VL" 的目录，所以真正生效的是下面这个函数。
# 两个节点（Qwen3_VQA / Qwen3_VL_BatchCache）都用它，保证下拉一致。


def scan_model_choices():
    """动态扫描 models/prompt_generator 下名字含 "Qwen3-VL" 的目录（上游行为）。

    包一层 try/except + 兜底的理由：`os.listdir` 在目录不存在时会抛
    FileNotFoundError，而 INPUT_TYPES 抛异常会让**整个节点注册失败**
    （用户没建 prompt_generator 目录就再也看不到节点了）。
    扫描为空时也回退——否则下拉变成空列表，ComfyUI 会直接报错。
    """
    try:
        prompt_generator_dir = os.path.join(folder_paths.models_dir, "prompt_generator")
        names = [n for n in os.listdir(prompt_generator_dir) if "Qwen3-VL" in n]
    except Exception as e:
        print(f"[Qwen3_VQA] 扫描 prompt_generator 目录失败，回退到内置模型列表：{e!r}")
        names = []
    return names or list(MODEL_CHOICES)


# eager 仍是第一项（新节点的默认值不变）。
# sage 由 sage_attention.py 注册，是**严格模式**：选了 sage 就一定走 sage，用不了会直接报错
# （不会静默退回 sdpa——那样测出来的成绩是假的）。比如没装 sageattention、或输入里有 padding，
# 都会抛出带指引的错误。
# 注意：本机没装 flash_attn，选 flash_attention_2 会直接报错。
ATTENTION_CHOICES = ["eager", "sage", "sdpa", "flash_attention_2"]

# 同一进程内对缓存文件的读写串行化，避免多节点并发写交错
_CACHE_LOCK = threading.Lock()


def _cache_file_for(image_path: str) -> str:
    return image_path + CACHE_SUFFIX


def _resolve_media_path(name: str) -> str:
    """解析媒体路径：绝对路径直通，其余走 ComfyUI 注解解析。

    与 util_nodes._resolve_media_path 同实现（无法安全互相 import：
    ComfyUI 核心也有顶层 nodes.py 模块，import 名字有歧义），保持一致。
    """
    name = (name or "").strip()
    is_abs = (
        (len(name) >= 2 and name[1] == ":")
        or name.startswith("\\\\")
        or name.startswith("/")
    )
    if is_abs:
        return os.path.abspath(name)
    return folder_paths.get_annotated_filepath(name)


def _next_id(image_path: str) -> str:
    today = datetime.datetime.now().strftime(ID_DATE_FMT)
    cache_file = _cache_file_for(image_path)
    max_n = 0
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = entry.get("id", "")
                if eid.startswith(today + "."):
                    try:
                        n = int(eid.rsplit(".", 1)[-1])
                    except ValueError:
                        continue
                    if n > max_n:
                        max_n = n
    return f"{today}.{max_n + 1:03d}"


def _read_entries(image_path: str):
    cache_file = _cache_file_for(image_path)
    entries = []
    if not os.path.exists(cache_file):
        return entries
    with _CACHE_LOCK:
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _find_entry(image_path: str, entry_id: str):
    for e in _read_entries(image_path):
        if e.get("id") == entry_id:
            return e
    return None


def _append_entry(image_path: str, entry: dict):
    cache_file = _cache_file_for(image_path)
    with _CACHE_LOCK:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _delete_entry(image_path: str, entry_id: str) -> bool:
    """从缓存文件中移除指定 id 的那一行；其余内容（含无法解析的行）原样保留。"""
    cache_file = _cache_file_for(image_path)
    if not os.path.exists(cache_file):
        return False
    with _CACHE_LOCK:
        with open(cache_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        kept = []
        removed = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                kept.append(line if line.endswith("\n") else line + "\n")
                continue
            if entry.get("id") == entry_id:
                removed = True
                continue
            kept.append(line if line.endswith("\n") else line + "\n")
        if not removed:
            return False
        tmp_file = cache_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp_file, cache_file)
    return True


def _make_entry(entry_id, model, text, output, seed, quantization, attention,
                temperature, max_new_tokens, min_pixels, max_pixels):
    return {
        "id": entry_id,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "text": text,  # 用户填的提示词原文
        "output": output,
        "seed": seed,
        "quantization": quantization,
        "attention": attention,
        "params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    }


def _entry_meta(entry: dict) -> dict:
    """列表接口用：不含完整 output，只给下拉/列表需要的信息。"""
    text = " ".join((entry.get("text") or "").split())
    if len(text) > SUMMARY_MAX:
        text = text[:SUMMARY_MAX] + "…"
    return {
        "id": entry.get("id", ""),
        "ts": entry.get("ts", ""),
        "model": entry.get("model", ""),
        "summary": text,
        "seed": entry.get("seed", None),
        "output_chars": len(entry.get("output") or ""),
        "params": entry.get("params", {}),
    }


# ---------------------------------------------------------------- 目录批量

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _scan_images(directory: str, recursive: bool = False):
    """列出目录下的图片文件（不含 xxx.png.json 这类缓存文件），按路径排序。"""
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(path)
    found.sort()
    return found


def _is_interrupted() -> bool:
    try:
        return bool(comfy.model_management.processing_interrupted())
    except Exception:
        return False


class _InterruptCheckCriteria:
    """挂在 model.generate() 的 stopping_criteria 上：每个生成步检查一次
    ComfyUI 的中断标志。

    ComfyUI 的中断信号原本只在节点之间生效——一次 generate 要跑几分钟，
    用户按了中断也得干等本节点结束。挂上它之后，中断请求会在下一个
    解码步（毫秒级）就被抛出，generate 立即终止。

    注意 InterruptProcessingException 继承自 BaseException，
    下面的 except Exception 不会把它吞掉，只兜住「没有 comfy 的
    独立运行/测试环境」的 AttributeError。"""

    def __call__(self, input_ids, scores, **kwargs):
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception:
            pass
        return False


# ComfyUI 的中断异常继承自 BaseException（不会被 except Exception 吞掉）。
# 独立运行/测试环境的桩里可能没有它，用 getattr 兜底。
_InterruptExc = getattr(comfy.model_management, "InterruptProcessingException", None)


def _release_runner(runner) -> None:
    """批量跑完后统一释放模型/处理器显存。"""
    try:
        if getattr(runner, "processor", None) is not None:
            del runner.processor
            runner.processor = None
        if getattr(runner, "model", None) is not None:
            del runner.model
            runner.model = None
        runner.current_model_id = None
        runner.current_quantization = None
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _estimate_model_bytes(path: str, quantization: str) -> int:
    """估算模型权重的显存占用（字节），用于判断加载前是否需要腾显存。

    safetensors 里存的是 bf16/fp16（2 字节/参数），bnb 量化按实际压缩比粗估
    并留余量：4bit ≈ 0.55 B/参数（0.35 倍），8bit ≈ 1.06 B/参数（0.65 倍）。
    估算不出（路径不存在 / 非本地目录）返回 0，调用方就当作"不干预"。
    """
    try:
        p = Path(path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.suffix == ".safetensors"]
        else:
            return 0
        total = sum(f.stat().st_size for f in files)
        if not total:
            return 0
        ratio = {"4bit": 0.35, "8bit": 0.65}.get(quantization, 1.0)
        return int(total * ratio)
    except Exception:
        return 0


# 本插件缓存了模型（未释放）的节点实例。ComfyUI 只能调度它自己管理的模型，
# 这里的 transformers 模型对它不可见——所以生命周期得自己管两头：
#   加载前：显存不够 -> 请 ComfyUI 卸载它的模型（见加载处 unload_all_models）；
#   让位后：新 prompt 不含 VQA 节点 -> 主动释放本插件模型，把显存还给 ComfyUI
#   （见文件尾部的 prompt 队列钩子）。
_VQA_INSTANCES = weakref.WeakSet()
_VQA_NODE_CLASSES = {"Qwen3_VQA", "Qwen3_VL_BatchCache"}


def _ensure_vram_for(need_bytes: int, label: str) -> None:
    """空闲显存不足以 {label} 时，请 ComfyUI 先卸载它管理的模型。

    ComfyUI 只管理它自己加载的模型（checkpoint/LoRA 走 model_management）；
    本节点经 transformers 直接加载，对它完全不可见。不主动请它让位的话，
    device_map="auto" / model.to(device) 就会因空闲显存不足而把权重塞进
    CPU（device_map 路径）或直接 OOM（to 路径）。走 unload_all_models()
    是 ComfyUI 的 smart memory 路径：权重留内存，下次快速重载。
    """
    if not need_bytes or not torch.cuda.is_available():
        return
    try:
        free = torch.cuda.mem_get_info()[0]
    except Exception:
        return
    if not free or free >= need_bytes * 1.1:
        return
    unloader = getattr(comfy.model_management, "unload_all_models", None)
    if not callable(unloader):
        return
    print(
        f"[Qwen3_VQA] 空闲显存 {free / 2**30:.1f} GB 不足以{label}"
        f"（约需 {need_bytes / 2**30:.1f} GB），先卸载 ComfyUI 管理的模型"
        f"（其权重会留在内存以便快速重载）"
    )
    try:
        unloader()
    except Exception as e:
        print(f"[Qwen3_VQA] 卸载 ComfyUI 模型失败（继续）：{e!r}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class Qwen3_VQA:
    def __init__(self):
        _VQA_INSTANCES.add(self)
        self._offloaded = False  # True = 模型整托管在内存里，用前先搬回显存
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.current_model_id = None  # Track the current model id
        self.current_quantization = None  # Track the current quantization
        # 下面这三个同样"只在加载时生效"，必须一起跟踪。
        # ComfyUI 会缓存节点实例（execution.py: caches.objects.get(unique_id)），
        # 漏掉它们的话，改了控件不会重载，新值会被静默忽略。
        self.current_attention = None
        self.current_min_pixels = None
        self.current_max_pixels = None

    def release_model(self, reason: str = "", offload: bool = False) -> bool:
        """释放/让位模型。offload=True 时优先整托管进内存（更快回归）。

        返回是否真的释放/让位了东西。
        - offload（让位）：model.to('cpu') 保留对象与 current_* 参数，
          下次 inference 只需一次 PCIe 搬回，不用重新解析 safetensors。
          仅 fp16/bf16 可行——bnb 量化的权重不支持搬设备；且模型带
          accelerate 多设备 dispatch（hf_device_map 混合设备）时 hook
          会被 .to 破坏，这两种情况退回彻底释放。
        - 彻底释放：del model/processor 并复位 current_*，下次完整重载。
        """
        if self.model is None and self.processor is None:
            return False
        if offload and self.model is not None and self.current_quantization == "none":
            _dm = getattr(self.model, "hf_device_map", None)
            _mixed = _dm is not None and (
                any(str(v) == "cpu" for v in _dm.values())
                or len(set(str(v) for v in _dm.values())) > 1
            )
            if not _mixed:
                try:
                    self.model.to("cpu")
                    self._offloaded = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    if reason:
                        print(f"[Qwen3_VQA] 模型已整托管进内存（{reason}），下次使用直接搬回显存")
                    return True
                except Exception as e:
                    print(f"[Qwen3_VQA] 整托管进内存失败，退回彻底释放：{e!r}")
        try:
            del self.model
            self.model = None
            del self.processor
            self.processor = None
            # current_* 一并复位：下次进来走完整重载路径
            self.current_model_id = None
            self.current_quantization = None
            self.current_attention = None
            self.current_min_pixels = None
            self.current_max_pixels = None
            self._offloaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if reason:
                print(f"[Qwen3_VQA] 已释放模型显存（{reason}）")
            return True
        except Exception as e:
            print(f"[Qwen3_VQA] 释放模型失败：{e!r}")
            return False

    def _restore_offloaded_model(self) -> bool:
        """把让位进内存的模型搬回显存。返回是否执行了搬回。"""
        if not self._offloaded or self.model is None:
            return False
        _ensure_vram_for(
            _estimate_model_bytes(self.model_checkpoint, self.current_quantization),
            "搬回模型",
        )
        self.model.to(self.device)
        self._offloaded = False
        print("[Qwen3_VQA] 模型已从内存搬回显存（跳过重载）")
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    scan_model_choices(),
                    {"default": "Qwen3-VL-4B-Instruct-FP8"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
                "attention": (ATTENTION_CHOICES,),
                "use_cache": ("BOOLEAN", {"default": True}),
                "image_path": ("STRING", {"default": ""}),
                "prompt_version": (
                    ["<new>"],
                    {"default": "<new>"},
                ),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def inference(
        self,
        text,
        model,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
        seed,
        quantization,
        use_cache,
        image_path,
        prompt_version,
        image=None,
        attention="eager",
    ):
        # 注意：下面第 300 行左右会用 apply_chat_template 的结果覆盖 text，这里先把提示词原文留一份
        user_text = text
        # image_path 是合并后的唯一媒体路径输入，同时兼任提示词缓存键：
        #   - str：Load Image Advanced / VideoLoader 的 path 输出，或手填的绝对路径；
        #   - list：MultiplePathsInput 的 content dict 列表（单条时取其中的路径当键，
        #           多条时没有单一文件可挂缓存，放弃缓存）。
        if isinstance(image_path, str) and image_path:
            cache_key = image_path
        elif (
            isinstance(image_path, list)
            and len(image_path) == 1
            and isinstance(image_path[0], dict)
        ):
            cache_key = image_path[0].get("image") or image_path[0].get("video")
        else:
            cache_key = None
        if use_cache and cache_key and prompt_version and prompt_version != "<new>":
            hit = _find_entry(cache_key, prompt_version)
            if hit is not None:
                return (hit.get("output", ""),)
        if seed != -1:
            torch.manual_seed(seed)

        # attention=sage 是严格模式：用不了就直接报错，绝不静默退回 sdpa。
        # 所以在下载/加载模型之前先确认 sageattention 真的在，避免白等几分钟才炸。
        # 这里直接 try import，不走 sageattn_available() 的缓存（用户可能中途才装上）。
        if attention == "sage":
            try:
                import sageattention  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "[Qwen3_VQA] attention=sage 需要 sageattention，但当前环境没有安装。\n"
                    "  请执行：pip install sageattention\n"
                    "  或者把节点的 attention 改成 sdpa / eager。"
                ) from e

        # 如果model名以abliterated结尾，则使用abliterated模型
        if "abliterated" in model:
            model_id = f"huihui-ai/{model}"
        else:
            model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        # 上次让位进内存的模型先搬回显存：current_* 没复位，下面的重载检查
        # 会因参数未变而跳过，直接用现成模型（比重新 from_pretrained 快得多）。
        self._restore_offloaded_model()

        # model_id / 量化 / 注意力实现 / 像素预算 变了就重载 processor 与模型。
        # 注意这些参数都只在"加载时"生效：attention 决定 attn_implementation，
        # min/max_pixels 决定 processor 的 smart_resize 预算。ComfyUI 会缓存节点实例，
        # 所以必须显式比较，否则改了控件这一轮不会生效（要重启才会）。
        if (
            self.current_model_id != model_id
            or self.current_quantization != quantization
            or self.current_attention != attention
            or self.current_min_pixels != min_pixels
            or self.current_max_pixels != max_pixels
            or self.processor is None
            or self.model is None
        ):
            if self.model is not None:
                print(
                    f"[Qwen3_VQA] 加载参数变化，重载模型 "
                    f"(model={model_id}, quant={quantization}, attn={attention}, "
                    f"pixels={min_pixels}~{max_pixels})"
                )
            self.current_model_id = model_id
            self.current_quantization = quantization
            self.current_attention = attention
            self.current_min_pixels = min_pixels
            self.current_max_pixels = max_pixels
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            # 空闲显存不足就先请 ComfyUI 让位（smart memory 权重留内存），
            # 避免 accelerate 把语言层 offload 到 CPU 导致推理慢一个数量级。
            _ensure_vram_for(
                _estimate_model_bytes(self.model_checkpoint, quantization), "加载模型"
            )

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )
            self._offloaded = False
            # device_map="auto" 按加载瞬间的空闲显存切分：显存不够就把语言层
            # offload 到 CPU，推理慢一个数量级。这里明确报出来，别让用户从
            # device_map 日志里自己猜。
            _dm = getattr(self.model, "hf_device_map", None)
            if _dm:
                _off = sum(1 for v in _dm.values() if str(v) in ("cpu", "disk"))
                if _off:
                    print(
                        f"[Qwen3_VQA] 警告：{_off}/{len(_dm)} 个模块被 offload 到 CPU/disk，"
                        f"推理会非常慢。建议：把节点量化设为 4bit，或先释放显存"
                        f"（卸掉 ComfyUI 里其它已加载的模型）再重跑本节点。"
                    )
            if _apply_vision_patch_fix is not None:
                try:
                    _apply_vision_patch_fix(self.model)
                except Exception as e:
                    print(f"[Qwen3_VQA] patch_embed 改写失败（不影响正确性，只是慢）：{e!r}")

        temp_path = None
        if image is not None:
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
            pil_image.save(temp_path)

        with torch.no_grad():
            if temp_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif image_path:
                # image_path 是合并后的唯一媒体来源：
                #   - str：按扩展名转成 image/video content dict；
                #   - list：MultiplePathsInput 的 content dict 列表（过滤识别失败的 None）。
                if isinstance(image_path, str):
                    ext = image_path.rsplit(".", 1)[-1].lower()
                    if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
                        content_items = [{"type": "image", "image": image_path}]
                    elif ext in ["mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"]:
                        content_items = [{"type": "video", "video": image_path}]
                    else:
                        raise ValueError(
                            f"[Qwen3_VQA] image_path 指向不支持的文件类型: {image_path}"
                        )
                else:
                    content_items = [c for c in image_path if c]
                content_items.append({"type": "text", "text": text})
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": content_items,
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            # Inference: Generation of the output
            # stopping_criteria：让 ComfyUI 的中断能在生成过程中（而不是
            # 等整个节点跑完）立刻生效。
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stopping_criteria=[_InterruptCheckCriteria()],
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                self.release_model("keep_model_loaded=False")

            output_text = result[0]

            if use_cache and cache_key:
                if prompt_version and prompt_version != "<new>":
                    entry_id = prompt_version
                    if _find_entry(cache_key, entry_id) is not None:
                        print(f"[Qwen3_VQA] cache id {entry_id} already exists for {cache_key}, skipping write")
                    else:
                        _append_entry(cache_key, _make_entry(
                            entry_id, model, user_text, output_text, seed,
                            quantization, attention, temperature, max_new_tokens,
                            min_pixels, max_pixels,
                        ))
                else:
                    entry_id = _next_id(cache_key)
                    _append_entry(cache_key, _make_entry(
                        entry_id, model, user_text, output_text, seed,
                        quantization, attention, temperature, max_new_tokens,
                        min_pixels, max_pixels,
                    ))

            return (output_text,)


class Qwen3_VL_BatchCache:
    """把一个目录下的图片逐张跑一遍 Qwen3-VL，结果按「一图一 json」写进缓存。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (scan_model_choices(), {"default": "Qwen3-VL-4B-Instruct-FP8"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "attention": (ATTENTION_CHOICES,),
                "recursive": ("BOOLEAN", {"default": False}),
                "skip_cached": ("BOOLEAN", {"default": True}),
                "force": ("BOOLEAN", {"default": False}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "seed": ("INT", {"default": -1}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"
    DESCRIPTION = """
批量把一个目录下的图片跑一遍 Qwen3-VL，结果写进「图片同名 .json」缓存。

- directory：要扫描的目录（只扫这一层，不递归，除非打开 recursive）
- text：所有图片共用的提示词
- skip_cached：该图已有缓存则跳过（增量补齐）
- force：打开后忽略 skip_cached，全部重新生成，按当天编号追加新条目
- limit：最多处理多少张，0 表示不限
- 节点上的「🔍 预扫描」按钮只统计数量，不会真的跑
"""

    def run(
        self,
        directory,
        text,
        model,
        quantization,
        attention,
        recursive,
        skip_cached,
        force,
        limit,
        seed,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
    ):
        directory = (directory or "").strip().strip('"')
        if not directory:
            return ("[Qwen3_VL_BatchCache] directory 为空，什么都没做。",)
        if not os.path.isdir(directory):
            return (f"[Qwen3_VL_BatchCache] 不是有效目录：{directory}",)

        images = _scan_images(directory, recursive)
        if limit and limit > 0:
            images = images[:limit]
        total = len(images)
        print(f"[Qwen3_VL_BatchCache] 目录 {directory} 命中 {total} 张图片（recursive={recursive}, limit={limit}）")

        released = []
        skipped = []
        failed = []
        runner = Qwen3_VQA()
        pbar = ProgressBar(total) if (ProgressBar and total) else None
        interrupted = False
        interrupt_exc = None

        for idx, img in enumerate(images, 1):
            if _is_interrupted():
                interrupted = True
                print(f"[Qwen3_VL_BatchCache] 收到中断，停在第 {idx} 张")
                break
            if skip_cached and not force and _read_entries(img):
                skipped.append(img)
                print(f"[Qwen3_VL_BatchCache] ({idx}/{total}) 跳过（已有缓存） {img}")
                if pbar:
                    pbar.update(1)
                continue
            try:
                runner.inference(
                    text=text,
                    model=model,
                    keep_model_loaded=True,  # 批量时模型只加载一次，最后统一释放
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    seed=seed,
                    quantization=quantization,
                    use_cache=True,
                    image_path=img,
                    prompt_version="<new>",
                    image=None,
                    attention=attention,
                )
                released.append(img)
                print(f"[Qwen3_VL_BatchCache] ({idx}/{total}) 完成 {img}")
            except BaseException as e:  # noqa: BLE001 - 必须区分 ComfyUI 的 BaseException 中断信号
                if _InterruptExc is not None and isinstance(e, _InterruptExc):
                    # 生成中途被中断：停止批量，先释放模型再原样抛回给 ComfyUI，
                    # 让它按标准中断流程收尾（而不是把中断记成「失败」继续跑）。
                    interrupted = True
                    interrupt_exc = e
                    print(f"[Qwen3_VL_BatchCache] ({idx}/{total}) 中断于 {img}")
                    break
                if not isinstance(e, Exception):
                    raise  # KeyboardInterrupt 等其它 BaseException 保持原语义
                failed.append((img, repr(e)))
                print(f"[Qwen3_VL_BatchCache] ({idx}/{total}) 失败 {img}: {e!r}")
            finally:
                if pbar:
                    pbar.update(1)

        _release_runner(runner)

        lines = [
            "[Qwen3_VL_BatchCache] 批量结束",
            f"目录        : {directory}",
            f"模式        : recursive={recursive}, skip_cached={skip_cached}, force={force}, limit={limit}",
            f"扫描到图片  : {total}",
            f"新生成      : {len(released)}",
            f"跳过(有缓存): {len(skipped)}",
            f"失败        : {len(failed)}",
        ]
        if interrupted:
            lines.insert(1, "!! 被中断，剩余图片未处理 !!")
        for path, err in failed:
            lines.append(f"  [失败] {path} -> {err}")
        report = "\n".join(lines)
        print(report)
        if interrupt_exc is not None:
            raise interrupt_exc  # 模型已释放，把中断交还给 ComfyUI 收尾
        return (report,)


if _HAS_SERVER:
    @PromptServer.instance.routes.get("/qwen3_vqa/cache/list")
    async def vqa_cache_list(request):
        image_path = request.query.get("image_path", "").strip()
        if not image_path:
            return web.json_response({"ids": [], "entries": [], "next_id": None})
        entries = _read_entries(image_path)
        metas = [_entry_meta(e) for e in entries]
        return web.json_response({
            "ids": [m["id"] for m in metas if m["id"]],
            "entries": metas,
            "next_id": _next_id(image_path),
        })

    @PromptServer.instance.routes.get("/qwen3_vqa/cache/get")
    async def vqa_cache_get(request):
        image_path = request.query.get("image_path", "").strip()
        entry_id = request.query.get("id", "").strip()
        if not image_path or not entry_id:
            return web.json_response(
                {"error": "image_path and id are required"}, status=400
            )
        entry = _find_entry(image_path, entry_id)
        if entry is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"entry": entry})

    @PromptServer.instance.routes.delete("/qwen3_vqa/cache/delete")
    async def vqa_cache_delete(request):
        image_path = request.query.get("image_path", "").strip()
        entry_id = request.query.get("id", "").strip()
        if not image_path or not entry_id:
            return web.json_response(
                {"error": "image_path and id are required"}, status=400
            )
        removed = _delete_entry(image_path, entry_id)
        return web.json_response({
            "ok": removed,
            "id": entry_id,
            "next_id": _next_id(image_path),
        })

    @PromptServer.instance.routes.get("/qwen3_vqa/resolve_path")
    async def vqa_resolve_path(request):
        """把 Load Image Advanced / VideoLoader 的文件名解析成绝对路径。

        给前端用：image_path 控件被转换成输入口后控件值是空的，
        前端要在不运行工作流的情况下沿连线反推路径，
        才能列出这张图已保存的 prompt（下拉栏）。"""
        name = request.query.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name is required"}, status=400)
        try:
            path = _resolve_media_path(name)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"path": path})

    @PromptServer.instance.routes.get("/qwen3_vqa/cache/index")
    async def vqa_cache_index(request):
        """全量缓存索引：前端下拉栏的兜底数据源（Lora Manager 同款思路）。

        prompt 下拉依赖 image_path，而 image_path 走连线时运行前拿不到值，
        连线反推也不总是可行（未知上游类型 / 连线刚拉上没触发刷新）。
        这里直接扫描 ComfyUI 的 input / output 根目录（+ 可选 roots 参数，
        分号分隔），把所有 sidecar 缓存列出来，前端不跑工作流也有选项。
        """
        roots_param = request.query.get("roots", "").strip()
        roots = [r.strip() for r in roots_param.split(";") if r.strip()]
        try:
            roots.insert(0, folder_paths.get_directory("input"))
            roots.append(folder_paths.get_directory("output"))
        except Exception:
            pass
        seen = set()
        images = []
        total_entries = 0
        truncated = False
        MAX_IMAGES = 3000
        for root in roots:
            if not root or root in seen or not os.path.isdir(root):
                continue
            seen.add(root)
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.endswith(CACHE_SUFFIX):
                        continue
                    cache_file = os.path.join(dirpath, fn)
                    image_path = cache_file[: -len(CACHE_SUFFIX)]
                    if not os.path.exists(image_path):
                        continue
                    metas = [_entry_meta(e) for e in _read_entries(image_path)]
                    metas = [m for m in metas if m.get("id")]
                    if not metas:
                        continue  # 不是本插件的缓存文件（普通 json）或还没有有效条目
                    total_entries += len(metas)
                    images.append({"path": image_path, "entries": metas})
                    if len(images) >= MAX_IMAGES:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break
        return web.json_response({
            "images": images,
            "total_images": len(images),
            "total_entries": total_entries,
            "truncated": truncated,
        })

    @PromptServer.instance.routes.get("/qwen3_vqa/batch/scan")
    async def vqa_batch_scan(request):
        """只读预扫描：统计目录里有多少图、多少张已有缓存，不写入任何东西。"""
        directory = request.query.get("directory", "").strip()
        recursive = request.query.get("recursive", "0") not in ("0", "", "false", "False")
        if not directory:
            # 前端拿不到目录时，允许按 image_path 反推所在目录
            probe = request.query.get("image_path", "").strip()
            if probe:
                directory = os.path.dirname(probe)
        if not directory:
            return web.json_response({"error": "directory is required"}, status=400)
        if not os.path.isdir(directory):
            return web.json_response(
                {"error": f"not a directory: {directory}"}, status=400
            )
        images = _scan_images(directory, recursive)
        files = []
        cached_count = 0
        for path in images:
            has_cache = bool(_read_entries(path))
            if has_cache:
                cached_count += 1
            files.append({
                "path": path,
                "name": os.path.basename(path),
                "cached": has_cache,
            })
        limit = 300
        return web.json_response({
            "directory": directory,
            "recursive": recursive,
            "total": len(images),
            "cached": cached_count,
            "pending": len(images) - cached_count,
            "files": files[:limit],
            "truncated": len(files) > limit,
        })


def _maybe_release_for_prompt(prompt) -> int:
    """新 prompt 不含 VQA 类节点时，释放本插件所有缓存实例的模型。返回释放个数。

    这是'反向让位'：ComfyUI 加载 SD 等模型前会清它自己的缓存，但看不见
    本插件的 16GB 模型，不主动让位的话它只能把自己降到 lowvram/流式加载。
    """
    if not isinstance(prompt, dict):
        return 0
    used = {v.get("class_type") for v in prompt.values() if isinstance(v, dict)}
    if used & _VQA_NODE_CLASSES:
        return 0
    released = 0
    for inst in list(_VQA_INSTANCES):
        try:
            # offload=True：fp16/bf16 模型整托管进内存，下次只搬回显存；
            # bnb 量化 / accelerate 混合设备的模型自动退回彻底释放。
            if inst.release_model("新任务不含 VQA 节点，为其它模型腾显存", offload=True):
                released += 1
        except Exception as e:
            print(f"[Qwen3_VQA] 让位释放失败（忽略）：{e!r}")
    return released


def _install_prompt_releaser() -> None:
    """包一层 PromptServer 的 prompt 队列：每次入队时检查要不要让位。"""
    try:
        q = PromptServer.instance.prompt_queue
    except Exception:
        return
    if q is None or getattr(q, "_qwen3_vqa_releaser", False):
        return
    orig_put = q.put

    def put(item):
        try:
            # 队列条目: (number, prompt_id, prompt, extra_data, outputs_to_execute)
            if isinstance(item, (tuple, list)) and len(item) > 2:
                n = _maybe_release_for_prompt(item[2])
                if n:
                    print(f"[Qwen3_VQA] 已为不含 VQA 的新任务释放 {n} 个实例的模型")
        except Exception as e:
            print(f"[Qwen3_VQA] prompt 队列钩子异常（忽略）：{e!r}")
        return orig_put(item)

    q.put = put
    q._qwen3_vqa_releaser = True


try:
    _install_prompt_releaser()
except Exception:
    # 测试环境 / PromptServer 尚未就绪时静默跳过，不影响节点加载
    pass

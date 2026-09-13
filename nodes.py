import os
import json
import datetime
import threading
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


class Qwen3_VQA:
    def __init__(self):
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
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
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
        source_path=None,
        image=None,
        attention="eager",
    ):
        # 注意：下面第 300 行左右会用 apply_chat_template 的结果覆盖 text，这里先把提示词原文留一份
        user_text = text
        if use_cache and image_path and prompt_version and prompt_version != "<new>":
            hit = _find_entry(image_path, prompt_version)
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

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
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
            if source_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_path:
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
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
                del self.processor  # release processor memory
                del self.model  # release model memory
                self.processor = None  # set processor to None
                self.model = None  # set model to None
                self.current_model_id = None
                self.current_quantization = None
                self.current_attention = None
                self.current_min_pixels = None
                self.current_max_pixels = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release GPU memory
                    torch.cuda.ipc_collect()

            output_text = result[0]

            if use_cache and image_path:
                if prompt_version and prompt_version != "<new>":
                    entry_id = prompt_version
                    if _find_entry(image_path, entry_id) is not None:
                        print(f"[Qwen3_VQA] cache id {entry_id} already exists for {image_path}, skipping write")
                    else:
                        _append_entry(image_path, _make_entry(
                            entry_id, model, user_text, output_text, seed,
                            quantization, attention, temperature, max_new_tokens,
                            min_pixels, max_pixels,
                        ))
                else:
                    entry_id = _next_id(image_path)
                    _append_entry(image_path, _make_entry(
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
                    source_path=[{"type": "image", "image": img}],
                    image=None,
                    attention=attention,
                )
                released.append(img)
                print(f"[Qwen3_VL_BatchCache] ({idx}/{total}) 完成 {img}")
            except Exception as e:
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

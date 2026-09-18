import hashlib
import os
import folder_paths
import numpy as np
import torch
import node_helpers
from PIL import Image, ImageOps, ImageSequence
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl


def _resolve_media_path(name: str) -> str:
    """解析媒体路径。

    绝对路径直通（web/js/drop_original_path.js 拖拽原始路径的场景——
    新版 folder_paths.get_annotated_filepath 会把不在 input 目录里的
    绝对路径当路径穿越拒绝掉）；其余（含 "foo.jpg [input]" 注解形式）
    仍走 ComfyUI 的解析。注意：nodes.py 里有一份同样实现（供 resolve_path
    路由用），两处保持一致，别只改一边。
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


class ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "path")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = _resolve_media_path(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, image_path)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = _resolve_media_path(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        try:
            ok = os.path.exists(_resolve_media_path(image))
        except ValueError as e:
            return str(e)
        if not ok:
            return "Invalid image file: {}".format(image)
        return True


class VideoLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {"file": (sorted(files), {"video_upload": True})},
        }

    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("VIDEO", "path")
    FUNCTION = "load_video"

    def load_video(self, file):
        video_path = _resolve_media_path(file)
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = _resolve_media_path(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        try:
            ok = os.path.exists(_resolve_media_path(file))
        except ValueError as e:
            return str(e)
        if not ok:
            return "Invalid video file: {}".format(file)
        return True


class VideoLoaderPath(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file": ("STRING", {"placeholder": "X://insert/path/here.mp4"}),
            },
        }

    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("VIDEO", "path")
    FUNCTION = "load_video"

    def load_video(self, file):
        video_path = _resolve_media_path(file)
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = _resolve_media_path(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        try:
            ok = os.path.exists(_resolve_media_path(file))
        except ValueError as e:
            return str(e)
        if not ok:
            return "Invalid video file: {}".format(file)
        return True

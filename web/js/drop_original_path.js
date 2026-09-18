// drop_original_path.js — 让 Load Image Advanced / VideoLoader / VideoLoaderPath
// 支持拖入文件时直接使用原始路径，跳过“复制到 ComfyUI\input”的上传流程。
//
// 背景：拖图进节点时核心前端会 POST /upload/image 把文件拷进 input，
// 控件值变成 input 里的拷贝路径。缓存 sidecar 写在原图旁边，于是
// image_path（= input 拷贝）找不到已保存的 prompt。
//
// 原理：ComfyUI 前端的 drop 分发在调用上传流程前先问节点——
//   dragover: if (!node.onDragOver?.(e)) 不算 dragOverNode
//   drop:     if (await node.onDragDrop?.(e)) return;  // 节点处理了就跳过核心逻辑
// 我们在 ImageLoader/VideoLoader/VideoLoaderPath 上实现这两个钩子：
// 通过 Desktop(Electron) 的 window.api.getPathForFile（preload 里的
// webUtils.getPathForFile）拿到原始绝对路径，直接填进控件。
// 拿不到原始路径的环境（普通浏览器）返回 false，自动回落到核心上传。
import { app } from "../../scripts/app.js";

const TARGET_TYPES = new Set(["ImageLoader", "VideoLoader", "VideoLoaderPath"]);
const MEDIA_EXT_RE = /\.(jpe?g|png|bmp|tiff|webp|gif|mp4|mkv|mov|avi|flv|wmv|webm|m4v)$/i;
const IMAGE_EXT_RE = /\.(jpe?g|png|bmp|tiff|webp|gif)$/i;

function resolveFilePath(file) {
    try {
        if (typeof file?.path === "string" && file.path) return file.path;
    } catch {}
    try {
        const p = window.api?.getPathForFile?.(file);
        if (typeof p === "string" && p) return p;
    } catch {}
    return null;
}

// dragover 阶段拿不到文件名，只能看 item 的 MIME 类型。Windows 上未注册
// 扩展名（.mkv/.flv 等）的文件 MIME 为空——用“全是空类型”兜底：拖工作流
// json / 图片链接时 MIME 都有明确值，不会被误伤。
function isMediaDrag(e) {
    const items = Array.from(e?.dataTransfer?.items || []).filter(
        (it) => it.kind === "file"
    );
    if (items.length === 0) return false;
    const types = items.map((it) => String(it.type || ""));
    if (types.some((t) => t.startsWith("image/") || t.startsWith("video/"))) {
        return true;
    }
    return types.every((t) => t === "");
}

app.registerExtension({
    name: "Comfyui_Qwen3-VL-Instruct.DropOriginalPath",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_TYPES.has(nodeData?.name)) return;
        const widgetName = nodeData.name === "ImageLoader" ? "image" : "file";

        nodeType.prototype.onDragOver = function (e) {
            return isMediaDrag(e);
        };

        nodeType.prototype.onDragDrop = async function (e) {
            const files = Array.from(e?.dataTransfer?.files || []);
            if (files.length !== 1) return false; // 多文件交给核心的多节点流程
            const file = files[0];
            if (!MEDIA_EXT_RE.test(file.name || "")) return false;

            const path = resolveFilePath(file);
            if (!path) return false; // 非 Desktop 环境 → 回落核心上传到 input

            const widget = this.widgets?.find((w) => w.name === widgetName);
            if (!widget) return false;
            widget.value = path;
            if (
                Array.isArray(widget.options?.values) &&
                !widget.options.values.includes(path)
            ) {
                widget.options.values.push(path);
            }
            try {
                await widget.callback?.(path);
            } catch (err) {
                console.warn("[DropOriginalPath] widget callback failed", err);
            }

            // 画布预览（仅图片）；视频预览交给运行时
            if (IMAGE_EXT_RE.test(file.name || "")) {
                try {
                    const img = new Image();
                    img.src = URL.createObjectURL(file);
                    this.imgs = [img];
                    this.setSizeForImage?.();
                    this.setDirtyCanvas?.(true, false);
                } catch (err) {
                    console.warn("[DropOriginalPath] preview failed", err);
                }
            }
            console.info(
                `[DropOriginalPath] ${nodeData.name}: 使用原始路径 ${path}（跳过复制到 input）`
            );
            return true;
        };
    },
});

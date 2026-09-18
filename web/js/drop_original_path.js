// drop_original_path.js — 让 Load Image Advanced / VideoLoader / VideoLoaderPath
// 支持拖入文件时直接使用原始路径，跳过“复制到 ComfyUI\input”的上传流程。
//
// 背景：拖图进节点时核心前端会 POST /upload/image 把文件拷进 input，
// 控件值变成 input 里的拷贝路径。缓存 sidecar 写在原图旁边，于是
// image_path（= input 拷贝）找不到已保存的 prompt。
//
// 原理：核心前端的 drop 分发顺序是——
//   dragover: 光标下节点的 onDragOver(e) 返回 true → 记为 dragOverNode
//   drop:     await dragOverNode.onDragDrop(e) 返回 true → 事件被节点吃掉，
//             核心“上传并自动创建 LoadImage 节点”的流程被跳过；
//             返回 false / 没有钩子 → 照旧走核心流程。
// 我们在 ImageLoader/VideoLoader/VideoLoaderPath 上实现这两个钩子：
// 通过 Desktop(Electron) 的 window.api.getPathForFile（preload 里的
// webUtils.getPathForFile）拿到原始绝对路径，直接填进控件。
//
// 重要行为保证：
// 1) 拿不到原始路径的环境（普通浏览器 / 旧版 Desktop preload）钩子一律返回
//    false，拖拽行为与未安装本扩展完全一致（拖到空白处/节点上都会自动建节点）。
// 2) 设置面板提供开关（Qwen3 VQA → 拖拽使用原始路径），关掉后钩子全部失效，
//    恢复核心默认行为；开关状态存 localStorage，不依赖 settings API。
import { app } from "/scripts/app.js";

const SETTING_ID = "Qwen3VL.DropOriginalPath.enabled";
const SETTING_LS_KEY = "Qwen3VL.DropOriginalPath.enabled";

function settingEnabled() {
    try {
        return localStorage.getItem(SETTING_LS_KEY) !== "0";
    } catch {
        return true;
    }
}

// 能否把拖入的 File 解析成磁盘上的原始路径。
// File.path 在 Electron >= 32 被移除，必须走 preload 暴露的 webUtils.getPathForFile。
function canResolveOriginalPath() {
    try {
        return typeof window.api?.getPathForFile === "function";
    } catch {
        return false;
    }
}

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

    // 设置面板开关：关掉 = 完全恢复核心默认的拖拽（自动建节点）行为
    settings: [
        {
            id: SETTING_ID,
            name: "Qwen3 VQA: 拖入文件到加载节点时使用原始路径（关闭则恢复拖拽自动建节点）",
            type: "boolean",
            defaultValue: true,
            onChange: (value) => {
                try {
                    localStorage.setItem(SETTING_LS_KEY, value ? "1" : "0");
                } catch {}
            },
        },
    ],

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_TYPES.has(nodeData?.name)) return;
        const widgetName = nodeData.name === "ImageLoader" ? "image" : "file";

        nodeType.prototype.onDragOver = function (e) {
            // 开关关闭或环境拿不到原始路径 → 一律不接管，核心行为原样保留
            if (!settingEnabled() || !canResolveOriginalPath()) return false;
            return isMediaDrag(e);
        };

        nodeType.prototype.onDragDrop = async function (e) {
            if (!settingEnabled()) return false;
            const files = Array.from(e?.dataTransfer?.files || []);
            if (files.length !== 1) return false; // 多文件交给核心的多节点流程
            const file = files[0];
            if (!MEDIA_EXT_RE.test(file.name || "")) return false;

            const path = resolveFilePath(file);
            if (!path) return false; // 非 Desktop 环境 → 回落核心上传/自动建节点

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
                `[DropOriginalPath] ${nodeData.name}: 使用原始路径 ${path}（跳过复制到 input）。` +
                `若想拖拽时自动新建节点，可在设置里关闭“Qwen3 VQA: 拖入文件到加载节点时使用原始路径”。`
            );
            return true;
        };
    },
});

import { app } from "../../scripts/app.js";

const NODE_NAME = "Qwen3_VQA";
const BATCH_NODE_NAME = "Qwen3_VL_BatchCache";
const NEW_VALUE = "<new>";
const NEW_LABEL = "<new>（新生成并保存）";

const enc = encodeURIComponent;

async function fetchJson(url, options) {
    const resp = await app.api.fetchApi(url, options);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText || ""}`.trim());
    return await resp.json();
}

const CacheAPI = {
    list(imagePath) {
        return fetchJson(`/qwen3_vqa/cache/list?image_path=${enc(imagePath)}`);
    },
    get(imagePath, id) {
        return fetchJson(
            `/qwen3_vqa/cache/get?image_path=${enc(imagePath)}&id=${enc(id)}`
        );
    },
    remove(imagePath, id) {
        return fetchJson(
            `/qwen3_vqa/cache/delete?image_path=${enc(imagePath)}&id=${enc(id)}`,
            { method: "DELETE" }
        );
    },
    scan(directory, recursive) {
        return fetchJson(
            `/qwen3_vqa/batch/scan?directory=${enc(directory)}&recursive=${recursive ? 1 : 0}`
        );
    },
    index() {
        return fetchJson(`/qwen3_vqa/cache/index`);
    },
};

function clip(text, max = 28) {
    const t = String(text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return "";
    return t.length > max ? t.slice(0, max) + "…" : t;
}

// image_path 控件被转换成输入口后，控件值是空的（真值只在运行时才有）。
// 要在不运行工作流的情况下刷新 prompt 下拉栏，就得沿连线反推路径：
//   ImageLoader / VideoLoader：读 image/file 控件的文件名，让后端解析成绝对路径
//   VideoLoaderPath：file 控件本身就是绝对路径
//   MultiplePathsInput：恰好填了一条 path_N 时用那条
//   中途的 Reroute / PrimitiveNode 会穿透
async function resolveImagePath(node) {
    const pathWidget = node.widgets?.find((w) => w.name === "image_path");
    const manual = String(pathWidget?.value || "").trim();
    if (manual) {
        // 绝对路径直接用；相对文件名（如 pasted\image (1).png）交给后端解析
        if (/^([a-zA-Z]:[\\/]|\\\\|\/)/.test(manual)) return manual;
        try {
            const data = await fetchJson(
                `/qwen3_vqa/resolve_path?name=${enc(manual)}`
            );
            if (data.path) return String(data.path);
        } catch (e) {
            console.warn("[Qwen3_VQA] resolve manual path failed", e);
        }
        return manual;
    }

    let input = node.inputs?.find((i) => i && i.name === "image_path");
    for (let hops = 0; hops < 10 && input; hops++) {
        const link = input.link != null ? app.graph.links[input.link] : null;
        if (!link) return "";
        const origin = app.graph.getNodeById(link.origin_id);
        if (!origin) return "";

        if (origin.type === "ImageLoader" || origin.type === "LoadImage" || origin.type === "VideoLoader") {
            const w = origin.widgets?.find((x) =>
                x.name === (origin.type === "VideoLoader" ? "file" : "image")
            );
            const name = String(w?.value || "").trim();
            if (!name) return "";
            const data = await fetchJson(
                `/qwen3_vqa/resolve_path?name=${enc(name)}`
            );
            return String(data.path || "");
        }
        if (origin.type === "VideoLoaderPath") {
            return String(
                origin.widgets?.find((x) => x.name === "file")?.value || ""
            ).trim();
        }
        if (origin.type === "MultiplePathsInput") {
            const vals = (origin.widgets || [])
                .filter((x) => /^path_\d+$/.test(x.name))
                .map((x) => String(x.value || "").trim())
                .filter(Boolean);
            return vals.length === 1 ? vals[0] : "";
        }
        if (origin.type === "PrimitiveNode") {
            return String(origin.widgets?.[0]?.value || "").trim();
        }
        if (origin.type === "Reroute") {
            input = origin.inputs?.[0];
            continue;
        }
        return ""; // 未知来源，解析不了
    }
    return "";
}

const STYLE_ID = "qwen3-vqa-cache-style";

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.qvqa-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99999;display:flex;align-items:center;justify-content:center;}
.qvqa-panel{background:#fff;color:#222;width:min(900px,92vw);max-height:84vh;border-radius:10px;display:flex;flex-direction:column;box-shadow:0 14px 44px rgba(0,0,0,.4);font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;font-size:13px;}
.qvqa-head{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid #e6e6e6;}
.qvqa-title{font-weight:600;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.qvqa-path{color:#888;font-size:12px;font-weight:400;}
.qvqa-btn{border:1px solid #d0d0d0;background:#f7f7f7;color:#333;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;}
.qvqa-btn:hover{background:#ededed;}
.qvqa-btn.danger{color:#b42318;border-color:#f0c2bd;background:#fff5f4;}
.qvqa-btn.danger:hover{background:#ffe8e5;}
.qvqa-body{padding:8px 16px 16px;overflow:auto;}
.qvqa-empty{color:#888;padding:24px 0;text-align:center;}
.qvqa-row{border:1px solid #ececec;border-radius:8px;padding:8px 10px;margin-top:8px;}
.qvqa-row .line1{display:flex;align-items:center;gap:8px;}
.qvqa-id{font-family:ui-monospace,Consolas,monospace;font-weight:600;}
.qvqa-meta{color:#999;font-size:11px;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.qvqa-sum{margin-top:4px;color:#444;}
.qvqa-detail{margin-top:8px;border-top:1px dashed #e0e0e0;padding-top:8px;}
.qvqa-detail label{display:block;color:#888;font-size:11px;margin-top:6px;}
.qvqa-detail pre{margin:2px 0 0;white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid #eee;border-radius:6px;padding:6px 8px;max-height:220px;overflow:auto;font-size:12px;}
.qvqa-stats{display:flex;gap:16px;flex-wrap:wrap;padding:6px 0 10px;border-bottom:1px solid #eee;margin-bottom:4px;}
.qvqa-stat b{font-size:16px;}
.qvqa-stat span{color:#888;font-size:12px;margin-left:4px;}
.qvqa-file{display:flex;align-items:center;gap:8px;padding:3px 2px;border-bottom:1px solid #f4f4f4;font-family:ui-monospace,Consolas,monospace;font-size:12px;}
.qvqa-file .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.qvqa-tag{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid transparent;}
.qvqa-tag.cached{background:#eaf6ec;color:#1c7c33;border-color:#c6e6cd;}
.qvqa-tag.pending{background:#fff4e5;color:#a35c00;border-color:#f2ddb8;}
.qvqa-hint{color:#888;font-size:12px;padding:6px 0 2px;}
`;
    document.head.appendChild(style);
}

function makeOverlay(titleText, subtitleText) {
    ensureStyle();
    const overlay = document.createElement("div");
    overlay.className = "qvqa-overlay";
    const panel = document.createElement("div");
    panel.className = "qvqa-panel";

    const head = document.createElement("div");
    head.className = "qvqa-head";
    const title = document.createElement("div");
    title.className = "qvqa-title";
    title.textContent = titleText;
    if (subtitleText) {
        const sub = document.createElement("span");
        sub.className = "qvqa-path";
        sub.textContent = `（${subtitleText}）`;
        title.appendChild(sub);
    }
    const actionSlot = document.createElement("div");
    actionSlot.style.display = "flex";
    actionSlot.style.gap = "8px";
    const closeBtn = document.createElement("button");
    closeBtn.className = "qvqa-btn";
    closeBtn.textContent = "关闭";
    head.append(title, actionSlot, closeBtn);

    const body = document.createElement("div");
    body.className = "qvqa-body";
    panel.append(head, body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const close = () => {
        overlay.remove();
        document.removeEventListener("keydown", onKey);
    };
    const onKey = (e) => {
        if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) close();
    });
    closeBtn.addEventListener("click", close);

    return { overlay, body, actionSlot, close };
}

function labelFor(value, metaMap) {
    if (value === NEW_VALUE) return NEW_LABEL;
    const meta = metaMap.get(value);
    if (!meta) return `${value} · (缓存缺失，运行将重建该条)`;
    const sum = clip(meta.summary);
    // 全量索引兜底时带出所属图片名，方便在同 id 条目间分辨
    const base = meta.image ? meta.image.split(/[\\/]/).pop() : "";
    const head = base ? `${value} · ${base}` : value;
    return sum ? `${head} · ${sum}` : head;
}

async function openManager(node) {
    const imagePath = await resolveImagePath(node).catch(() => "");
    if (!imagePath) {
        alert(
            "无法确定图片路径：请手填 image_path，" +
            "或把它连到 Load Image Advanced / VideoLoader 等节点并选好文件。"
        );
        return;
    }

    const { body, actionSlot } = makeOverlay("VQA 提示词缓存", imagePath);

    const reloadBtn = document.createElement("button");
    reloadBtn.className = "qvqa-btn";
    reloadBtn.textContent = "刷新";
    actionSlot.appendChild(reloadBtn);

    async function afterChange() {
        try {
            node.__vqaRefresh?.();
        } catch (e) {
            console.warn("[Qwen3_VQA] refresh failed", e);
        }
    }

    async function showDetail(container, id) {
        if (container.dataset.open === "1") {
            container.innerHTML = "";
            container.dataset.open = "0";
            return;
        }
        container.dataset.open = "1";
        container.innerHTML = "";
        const loading = document.createElement("div");
        loading.className = "qvqa-sum";
        loading.textContent = "读取中…";
        container.appendChild(loading);
        try {
            const data = await CacheAPI.get(imagePath, id);
            const entry = data.entry || {};
            container.innerHTML = "";

            const mk = (label, value) => {
                const lab = document.createElement("label");
                lab.textContent = label;
                const pre = document.createElement("pre");
                pre.textContent = value;
                container.append(lab, pre);
            };

            mk("提示词", entry.text || "(空)");
            mk("生成结果", entry.output || "(空)");
            const p = entry.params || {};
            const parts = [];
            if (entry.model) parts.push(`model=${entry.model}`);
            if (entry.ts) parts.push(`ts=${entry.ts}`);
            if (entry.seed !== undefined && entry.seed !== null) parts.push(`seed=${entry.seed}`);
            if (entry.quantization) parts.push(`quant=${entry.quantization}`);
            if (entry.attention) parts.push(`attn=${entry.attention}`);
            if (p.temperature !== undefined) parts.push(`temp=${p.temperature}`);
            if (p.max_new_tokens !== undefined) parts.push(`max_new_tokens=${p.max_new_tokens}`);
            if (p.min_pixels !== undefined) parts.push(`min_pixels=${p.min_pixels}`);
            if (p.max_pixels !== undefined) parts.push(`max_pixels=${p.max_pixels}`);
            mk("参数", parts.join("  "));
        } catch (e) {
            container.innerHTML = "";
            const err = document.createElement("div");
            err.className = "qvqa-sum";
            err.textContent = `读取失败：${e.message}`;
            container.appendChild(err);
        }
    }

    function render(entries) {
        body.innerHTML = "";
        if (!entries?.length) {
            const empty = document.createElement("div");
            empty.className = "qvqa-empty";
            empty.textContent = "还没有缓存条目。用 <new> 跑一次就会自动写入。";
            body.appendChild(empty);
            return;
        }
        for (const meta of entries) {
            const row = document.createElement("div");
            row.className = "qvqa-row";

            const line1 = document.createElement("div");
            line1.className = "line1";
            const idEl = document.createElement("span");
            idEl.className = "qvqa-id";
            idEl.textContent = meta.id || "(无 id)";
            const metaEl = document.createElement("span");
            metaEl.className = "qvqa-meta";
            metaEl.textContent = `${meta.ts || "-"} · ${meta.model || "-"} · ${meta.output_chars ?? 0} 字`;
            const viewBtn = document.createElement("button");
            viewBtn.className = "qvqa-btn";
            viewBtn.textContent = "查看";
            const delBtn = document.createElement("button");
            delBtn.className = "qvqa-btn danger";
            delBtn.textContent = "删除";
            line1.append(idEl, metaEl, viewBtn, delBtn);

            const sum = document.createElement("div");
            sum.className = "qvqa-sum";
            sum.textContent = meta.summary || "(这条没有存提示词)";

            const detail = document.createElement("div");
            detail.className = "qvqa-detail";
            detail.dataset.open = "0";

            viewBtn.addEventListener("click", () => showDetail(detail, meta.id));
            delBtn.addEventListener("click", async () => {
                if (!confirm(`确认删除缓存条目 ${meta.id} ？\n（只改 .json 文件，不动图片）`)) return;
                delBtn.disabled = true;
                try {
                    await CacheAPI.remove(imagePath, meta.id);
                    await load();
                    await afterChange();
                } catch (e) {
                    alert(`删除失败：${e.message}`);
                    delBtn.disabled = false;
                }
            });

            row.append(line1, sum, detail);
            body.appendChild(row);
        }
    }

    async function load() {
        body.innerHTML = "";
        const loading = document.createElement("div");
        loading.className = "qvqa-empty";
        loading.textContent = "加载中…";
        body.appendChild(loading);
        try {
            const data = await CacheAPI.list(imagePath);
            render(data.entries || []);
        } catch (e) {
            body.innerHTML = "";
            const err = document.createElement("div");
            err.className = "qvqa-empty";
            err.textContent = `读取失败：${e.message}`;
            body.appendChild(err);
        }
    }

    reloadBtn.addEventListener("click", load);
    load();
}

function openScanner(node) {
    const widget = (name) => node.widgets?.find((w) => w.name === name);
    const directory = String(widget("directory")?.value || "").trim();
    if (!directory) {
        alert("请先在节点上填写 directory（要扫描的目录）");
        return;
    }
    const recursive = !!widget("recursive")?.value;
    const skipCached = !!widget("skip_cached")?.value;
    const force = !!widget("force")?.value;
    const limit = Number(widget("limit")?.value || 0);

    const { body, actionSlot } = makeOverlay("目录预扫描（只统计，不会生成）", directory);

    const reloadBtn = document.createElement("button");
    reloadBtn.className = "qvqa-btn";
    reloadBtn.textContent = "重新扫描";
    actionSlot.appendChild(reloadBtn);

    const stat = (num, text) => {
        const d = document.createElement("div");
        d.className = "qvqa-stat";
        const b = document.createElement("b");
        b.textContent = String(num);
        const s = document.createElement("span");
        s.textContent = text;
        d.append(b, s);
        return d;
    };

    async function load() {
        body.innerHTML = "";
        const loading = document.createElement("div");
        loading.className = "qvqa-empty";
        loading.textContent = "扫描中…";
        body.appendChild(loading);
        try {
            const data = await CacheAPI.scan(directory, recursive);
            body.innerHTML = "";

            const stats = document.createElement("div");
            stats.className = "qvqa-stats";
            stats.append(
                stat(data.total, "张图片"),
                stat(data.cached, "已有缓存"),
                stat(data.pending, "待生成")
            );
            body.appendChild(stats);

            const willRun = skipCached && !force ? data.pending : data.total;
            const hint = document.createElement("div");
            hint.className = "qvqa-hint";
            const shown = limit > 0 ? Math.min(willRun, limit) : willRun;
            hint.textContent =
                skipCached && !force
                    ? `按当前设置（skip_cached 开、force 关），点「运行」会新生成 ${shown} 张，跳过 ${data.cached} 张。` +
                      (limit > 0 ? `（limit=${limit}）` : "")
                    : `按当前设置（${force ? "force 开" : "skip_cached 关"}），点「运行」会重新生成全部 ${shown} 张。` +
                      (limit > 0 ? `（limit=${limit}）` : "");
            body.appendChild(hint);

            const list = document.createElement("div");
            list.style.marginTop = "6px";
            for (const f of data.files || []) {
                const row = document.createElement("div");
                row.className = "qvqa-file";
                const nm = document.createElement("span");
                nm.className = "nm";
                nm.textContent = f.path;
                nm.title = f.path;
                const tag = document.createElement("span");
                tag.className = `qvqa-tag ${f.cached ? "cached" : "pending"}`;
                tag.textContent = f.cached ? "已缓存" : "待生成";
                row.append(nm, tag);
                list.appendChild(row);
            }
            body.appendChild(list);

            if (data.truncated) {
                const more = document.createElement("div");
                more.className = "qvqa-hint";
                more.textContent = `（文件列表只显示前 ${data.files.length} 个）`;
                body.appendChild(more);
            }
        } catch (e) {
            body.innerHTML = "";
            const err = document.createElement("div");
            err.className = "qvqa-empty";
            err.textContent = `扫描失败：${e.message}`;
            body.appendChild(err);
        }
    }

    reloadBtn.addEventListener("click", load);
    load();
}

function registerVqaCache(nodeType) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

        const node = this;
        const pathWidget = node.widgets?.find((w) => w.name === "image_path");
        const versionWidget = node.widgets?.find((w) => w.name === "prompt_version");
        const useCacheWidget = node.widgets?.find((w) => w.name === "use_cache");
        if (!pathWidget || !versionWidget) return;

        // id -> 元信息，用于给下拉生成 "id · 提示词摘要" 这样的标签
        const metaMap = new Map();

        // 函数式 values（动态下拉的标准手段）：前端 ComboWidget 每次点开下拉
        // 都会调 getValues()，values 是函数就现场调用。返回缓存列表让菜单
        // 立即显示，同时排队一次异步 refresh，下次打开就是最新数据。
        // 这样无论路径从哪条路来（连线/手填/外部新增缓存），打开下拉必然是新的。
        const valuesRef = { list: [NEW_VALUE] };
        let refreshQueued = false;
        versionWidget.options.values = () => {
            if (!refreshQueued) {
                refreshQueued = true;
                setTimeout(() => {
                    refreshQueued = false;
                    refresh();
                }, 0);
            }
            return valuesRef.list;
        };

        function applyValues(values) {
            valuesRef.list.length = 0;
            for (const v of values) valuesRef.list.push(v);
        }

        function applyLabelMapper() {
            versionWidget.options.getOptionLabel = (value) => labelFor(value, metaMap);
        }

        function resetDropdown() {
            metaMap.clear();
            applyValues([NEW_VALUE]);
            applyLabelMapper();
            if (versionWidget.value !== NEW_VALUE) versionWidget.value = NEW_VALUE;
        }

        // 全量索引兜底：路径解析不出来，或解析出来但这张图没有任何缓存条目。
        // 选项标签带所属图片名；若 image_path 还是自由控件（没连线、没手填），
        // 选中某条时会顺手把它的图片路径填进去。
        async function fillFromIndex() {
            try {
                const data = await CacheAPI.index();
                metaMap.clear();
                const values = [NEW_VALUE];
                for (const img of data.images || []) {
                    for (const meta of img.entries || []) {
                        if (!meta.id || metaMap.has(meta.id)) continue;
                        metaMap.set(meta.id, { ...meta, image: img.path });
                        values.push(meta.id);
                    }
                }
                const current = versionWidget.value;
                if (current && current !== NEW_VALUE && !values.includes(current)) {
                    values.push(current);
                }
                applyValues(values);
                applyLabelMapper();
                if (!values.includes(versionWidget.value)) {
                    versionWidget.value = NEW_VALUE;
                }
            } catch (e) {
                console.warn("[Qwen3_VQA] cache index fetch failed", e);
                resetDropdown();
            }
        }

        const refresh = async () => {
            if (!useCacheWidget?.value) {
                resetDropdown();
                return;
            }

            let imagePath = "";
            try {
                // 手填控件值，或沿连线反推（Load Image Advanced 等）
                imagePath = await resolveImagePath(node);
            } catch (e) {
                console.warn("[Qwen3_VQA] resolve image_path failed", e);
            }

            if (!imagePath) {
                // 路径解析不出来（未知上游类型 / 连线刚拉上 / 没连）→
                // 兜底走后端全量缓存索引，不运行工作流也有选项可选。
                await fillFromIndex();
                return;
            }

            try {
                const data = await CacheAPI.list(imagePath);
                const ids = (data.ids || []).filter(Boolean);
                if (!ids.length) {
                    // 这张图本地没有缓存条目（比如拖入的拷贝、sidecar 在别处）
                    // → 也走全量索引兜底，让下拉栏始终有内容可选
                    await fillFromIndex();
                    return;
                }
                metaMap.clear();
                for (const meta of data.entries || []) {
                    if (meta.id) metaMap.set(meta.id, meta);
                }
                const values = [NEW_VALUE, ...ids];
                // 当前值不在列表里也保留（比如换了图 / json 被移走），不做静默替换
                const current = versionWidget.value;
                if (current && current !== NEW_VALUE && !values.includes(current)) {
                    values.push(current);
                }
                applyValues(values);
                applyLabelMapper();
                if (!values.includes(versionWidget.value)) {
                    versionWidget.value = NEW_VALUE;
                }
            } catch (e) {
                console.warn("[Qwen3_VQA] cache list fetch failed", e);
            }
        };

        const refreshBtn = node.addWidget("button", "🔄 刷新 VQA 缓存", null, () =>
            refresh()
        );
        if (refreshBtn) refreshBtn.serialize = false;

        const managerBtn = node.addWidget("button", "🗂 缓存管理", null, () =>
            openManager(node)
        );
        if (managerBtn) managerBtn.serialize = false;

        node.__vqaRefresh = refresh;

        const wrap = (target) => {
            if (!target) return;
            const orig = target.callback;
            target.callback = function () {
                const result = orig?.apply(this, arguments);
                refresh();
                return result;
            };
        };
        wrap(pathWidget);
        wrap(useCacheWidget);

        // 从全量索引里选中某条时：如果 image_path 还是自由控件（没连线、没手填），
        // 把该条目所属图片的路径填进去，保证运行时缓存能命中
        const origVersionCb = versionWidget.callback;
        versionWidget.callback = function () {
            const result = origVersionCb?.apply(this, arguments);
            try {
                const meta = metaMap.get(versionWidget.value);
                const linked = node.inputs?.some(
                    (i) => i && i.name === "image_path" && i.link != null
                );
                if (meta?.image && !linked && pathWidget && !String(pathWidget.value || "").trim()) {
                    pathWidget.value = meta.image;
                }
            } catch (e) {
                console.warn("[Qwen3_VQA] fill image_path from index failed", e);
            }
            return result;
        };

        applyLabelMapper();
        refresh();
    };

    // 连线变化（拉上/拔掉 image_path 的上游）时自动刷新下拉栏——
    // 之前只有手点刷新按钮或换图才会刷，连线刚拉上时下拉永远是 <new>
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
        const result = origOnConnectionsChange?.apply(this, arguments);
        setTimeout(() => {
            try {
                this.__vqaRefresh?.();
            } catch (e) {
                console.warn("[Qwen3_VQA] refresh on connection change failed", e);
            }
        }, 0);
        return result;
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = origOnConfigure?.apply(this, arguments);
        setTimeout(() => {
            try {
                this.__vqaRefresh?.();
            } catch (e) {
                console.warn("[Qwen3_VQA] refresh on configure failed", e);
            }
        }, 0);
        return result;
    };
}

function registerBatchScanner(nodeType) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

        const scanBtn = this.addWidget("button", "🔍 预扫描目录", null, () =>
            openScanner(this)
        );
        if (scanBtn) scanBtn.serialize = false;
    };
}

// 在加载节点（ImageLoader/VideoLoader/VideoLoaderPath）上换文件时，
// 把 path 输出连着的下游 VQA 节点的 prompt 下拉栏刷一遍
function registerLoaderPropagate(nodeType, widgetName) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
        const node = this;
        const w = node.widgets?.find((x) => x.name === widgetName);
        if (!w) return;
        const origCb = w.callback;
        w.callback = function () {
            const result = origCb?.apply(this, arguments);
            setTimeout(() => {
                for (const out of node.outputs || []) {
                    if (!out?.links) continue;
                    for (const lid of out.links) {
                        const link = app.graph.links[lid];
                        if (!link) continue;
                        const target = app.graph.getNodeById(link.target_id);
                        try {
                            target?.__vqaRefresh?.();
                        } catch (e) {
                            console.warn("[Qwen3_VQA] downstream refresh failed", e);
                        }
                    }
                }
            }, 0);
            return result;
        };
    };
}

app.registerExtension({
    name: "Comfyui_Qwen3-VL-Instruct.VQACache",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === NODE_NAME) {
            registerVqaCache(nodeType);
        } else if (nodeData?.name === BATCH_NODE_NAME) {
            registerBatchScanner(nodeType);
        } else if (nodeData?.name === "ImageLoader") {
            registerLoaderPropagate(nodeType, "image");
        } else if (nodeData?.name === "VideoLoader" || nodeData?.name === "VideoLoaderPath") {
            registerLoaderPropagate(nodeType, "file");
        }
    },
});

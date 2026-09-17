"""离线验证 Qwen3-VL 缓存 / 目录批量 / 注意力选项（用桩替换 torch、transformers 等重依赖）。

不需要 GPU、不需要装模型，任何 python 都能跑：

    python tests/test_cache_and_batch.py

它只测插件自己的逻辑（目录扫描、JSONL 缓存读写、批量循环、HTTP 接口、下拉选项），
不测真实推理。sage 走的真实路径由 tests/test_sage_attention.py 用 GPU 验证。
"""
import sys, os, io, types, json, asyncio, tempfile, shutil, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)


def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------- 桩
torch = mod('torch', no_grad=lambda: contextlib.nullcontext(),
            manual_seed=lambda s: None, bfloat16='bf16', float16='fp16')
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None,
                                   ipc_collect=lambda: None, get_device_capability=lambda d: (0, 0))
torch.jit = types.SimpleNamespace(is_tracing=lambda: False)
tv = mod('torchvision')
tv.transforms = mod('torchvision.transforms', ToPILImage=lambda: None)

tf = mod('transformers', Qwen3VLForConditionalGeneration=None, AutoProcessor=None,
         BitsAndBytesConfig=None)
tf.integrations = mod('transformers.integrations')
tf.integrations.sdpa_attention = mod(
    'transformers.integrations.sdpa_attention',
    repeat_kv=lambda x, n: x,
    sdpa_attention_forward=lambda *a, **k: (None, None))
tf.modeling_utils = mod('transformers.modeling_utils')
tf.modeling_utils.AttentionInterface = type('AttentionInterface', (), {
    '_global_mapping': {},
    'register': classmethod(lambda cls, key, value: cls._global_mapping.update({key: value})),
    'valid_keys': lambda self: [],
})

# sageattention 也要打桩。nodes.py 在 attention=='sage' 时会在「下载/加载模型之前」
# 先 try import 它（严格模式：用不了要立刻报错，而不是跑到前向才炸），
# 这里 fake 的 torch 撑不起真的 sageattention，所以必须自己塞一个。
# 想测「没装 sageattention 时该报错」的用例，把 sys.modules 里这条临时设成 None。
mod('sageattention', sageattn=lambda *a, **k: None)

# comfy.model_management 桩：模拟真实的中断语义
# （InterruptProcessingException 继承 BaseException；throw_... 抛出时消费标志）
class _InterruptProcessingException(BaseException):
    pass


mm = mod('comfy.model_management', get_torch_device=lambda: 'cpu')
mm.InterruptProcessingException = _InterruptProcessingException
mm._flag = {'v': False}
mm.processing_interrupted = lambda: mm._flag['v']
mm.interrupt_current_processing = lambda v=True: mm._flag.__setitem__('v', bool(v))


def _throw_if_processing_interrupted():
    if mm._flag['v']:
        mm._flag['v'] = False
        raise _InterruptProcessingException()


mm.throw_exception_if_processing_interrupted = _throw_if_processing_interrupted
comfy = mod('comfy', model_management=mm)


class _PBar:
    def __init__(self, total):
        self.total, self.n = total, 0

    def update(self, k=1):
        self.n += k


comfy.utils = mod('comfy.utils', ProgressBar=_PBar)
mod('qwen_vl_utils', process_vision_info=lambda m: (None, None))
mod('folder_paths', models_dir=tempfile.gettempdir(), temp_directory=tempfile.gettempdir())

REG = {}


class _Routes:
    def get(self, path):
        return lambda fn: REG.setdefault(('GET', path), fn)

    def delete(self, path):
        return lambda fn: REG.setdefault(('DELETE', path), fn)


mod('server', PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes())))


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status


web = types.SimpleNamespace(json_response=lambda payload, status=200: _Resp(payload, status))
mod('aiohttp', web=web)

sys.path.insert(0, PLUGIN)
nodes = __import__('nodes')

OK = []


def chk(name, cond, extra=''):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} | {name} {extra}")


# ================================================================ 1. 目录扫描
root = os.path.join(tempfile.gettempdir(), 'qvqa_batch_dir')
if os.path.isdir(root):
    shutil.rmtree(root)
os.makedirs(os.path.join(root, 'sub'), exist_ok=True)
for name in ['a.png', 'b.JPG', 'c.txt', 'd.webp', 'note.md']:
    open(os.path.join(root, name), 'wb').write(b'x')
open(os.path.join(root, 'sub', 'e.png'), 'wb').write(b'x')
open(os.path.join(root, 'a.png.json'), 'w', encoding='utf-8').write('{"id":"2099.01.01.001","output":"old"}\n')

print('=== 1. 目录扫描 ===')
flat = nodes._scan_images(root, recursive=False)
chk('只扫当前层 => 3 张，且排除 .txt/.md/.json', len(flat) == 3, [os.path.basename(p) for p in flat])
chk('扩展名大小写兼容 (b.JPG)', any(p.endswith('b.JPG') for p in flat))
chk('不把 a.png.json 当图片', not any(p.endswith('.json') for p in flat))
chk('递归 => 4 张', len(nodes._scan_images(root, recursive=True)) == 4)
chk('扫描结果已排序', flat == sorted(flat))
chk('目录不存在 => 空', nodes._scan_images(os.path.join(root, 'nope')) == [])

# ================================================================ 2. 缓存读写
print('\n=== 2. 单图缓存读写 ===')
img = os.path.join(root, 'a.png')
cache = nodes._cache_file_for(img)
chk('缓存文件贴在图片旁边', cache == img + '.json', cache)
eid = nodes._next_id(img)
nodes._append_entry(img, nodes._make_entry(eid, 'm1', '提示词A', 'OUT-A', 7, 'none', 'eager', 0.7, 2048, 100, 200))
chk('二次写入得到新 id', nodes._next_id(img) != eid)
chk('_find_entry 命中', nodes._find_entry(img, eid)['output'] == 'OUT-A')
chk('_entry_meta 带摘要', nodes._entry_meta(nodes._find_entry(img, eid))['summary'] == '提示词A')
chk('_delete_entry 生效', nodes._delete_entry(img, eid) is True)
chk('重复删除返回 False', nodes._delete_entry(img, eid) is False)
with open(cache, 'a', encoding='utf-8') as f:
    f.write('这不是合法JSON\n')
eid2 = nodes._next_id(img)
nodes._append_entry(img, nodes._make_entry(eid2, 'm1', '提示词B', 'OUT-B', 1, 'none', 'eager', 0.7, 1, 1, 1))
nodes._delete_entry(img, eid2)
chk('删除时保留解析不了的行', '这不是合法JSON' in open(cache, encoding='utf-8').read())
os.remove(cache)

# ================================================================ 3. 批量节点
print('\n=== 3. 批量节点 run() ===')
calls = []
_REAL_INFERENCE = nodes.Qwen3_VQA.inference  # 第 3 节会把它换掉，第 6 节要还原


def fake_inference(self, text, model, keep_model_loaded, temperature, max_new_tokens,
                   min_pixels, max_pixels, seed, quantization, use_cache, image_path,
                   prompt_version, image=None, attention='eager'):
    calls.append(dict(image_path=image_path, text=text,
                      prompt_version=prompt_version, keep_model_loaded=keep_model_loaded,
                      attention=attention))
    e = nodes._next_id(image_path)
    nodes._append_entry(image_path, nodes._make_entry(
        e, model, text, f'OUT-{os.path.basename(image_path)}', seed, quantization,
        attention, temperature, max_new_tokens, min_pixels, max_pixels))
    return (f'OUT-{os.path.basename(image_path)}',)


nodes.Qwen3_VQA.inference = fake_inference
batch = nodes.Qwen3_VL_BatchCache()

kw = dict(directory=root, text='描述这张图', model='Qwen3-VL-4B-Instruct-FP8',
          quantization='none', attention='eager', recursive=False, skip_cached=True,
          force=False, limit=0, seed=-1, temperature=0.7, max_new_tokens=2048,
          min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

report = batch.run(**kw)[0]
chk('第一轮 3 张全部生成',
    [os.path.basename(c['image_path']) for c in calls] == ['a.png', 'b.JPG', 'd.webp'],
    ' | ' + report.replace('\n', ' / '))
chk('批量把图片绝对路径传给 image_path', os.path.isabs(calls[0]['image_path']))
chk('prompt_version 用 <new>（自动编号）', calls[0]['prompt_version'] == '<new>')
chk('批量期间 keep_model_loaded=True', all(c['keep_model_loaded'] for c in calls))
chk('提示词为统一提示词', calls[0]['text'] == '描述这张图')
chk('每张图都写出缓存', all(len(nodes._read_entries(c['image_path'])) == 1 for c in calls))

calls.clear()
report = batch.run(**kw)[0]
chk('第二轮全部跳过（增量）', len(calls) == 0 and '新生成      : 0' in report)

calls.clear()
report = batch.run(**{**kw, 'force': True})[0]
chk('force 打开后重新生成', len(calls) == 3)
chk('force 是追加而不是覆盖', len(nodes._read_entries(img)) == 2)

calls.clear()
batch.run(**{**kw, 'limit': 1, 'force': True})
chk('limit=1 只处理一张', len(calls) == 1)

calls.clear()
batch.run(**{**kw, 'recursive': True, 'force': True})
chk('recursive 带上子目录', len(calls) == 4, [os.path.basename(c['image_path']) for c in calls])

chk('空目录参数被拦下', 'directory 为空' in batch.run(**{**kw, 'directory': ''})[0])
chk('非法目录被拦下', '不是有效目录' in batch.run(**{**kw, 'directory': root + '_nope'})[0])

good = fake_inference


def flaky(self, **kwargs):
    if kwargs['image_path'].endswith('b.JPG'):
        raise RuntimeError('模拟推理失败')
    return good(self, **kwargs)


nodes.Qwen3_VQA.inference = flaky
report = batch.run(**{**kw, 'force': True})[0]
chk('单张失败会记录并继续', '失败        : 1' in report and '模拟推理失败' in report)
nodes.Qwen3_VQA.inference = fake_inference

# ================================================================ 3.5 生成中断
print('\n=== 3.5 生成中断（stopping criteria / BatchCache 中断处理） ===')

# 单独测 criteria：无中断 -> False；有中断 -> 抛 BaseException 并消费标志
crit = nodes._InterruptCheckCriteria()
chk('无中断时 criteria 返回 False', crit([1, 2], None) is False)
mm._flag['v'] = True
raised_crit = None
try:
    crit([1, 2], None)
except BaseException as e:
    raised_crit = e
chk('criteria 检测到中断抛 BaseException', isinstance(raised_crit, _InterruptProcessingException), repr(raised_crit))
chk('抛出时消费了中断标志', mm._flag['v'] is False)

# 批量场景：b.JPG 生成到一半被中断
calls.clear()


def interrupting_inference(self, **kwargs):
    img = kwargs['image_path']
    calls.append(os.path.basename(img))
    if img.endswith('b.JPG'):
        raise _InterruptProcessingException()  # 模拟 stopping criteria 在生成中触发
    e = nodes._next_id(img)
    nodes._append_entry(img, nodes._make_entry(
        e, 'm', 't', f'OUT-{os.path.basename(img)}', -1, 'none', 'eager', 0.7, 1, 1, 1))
    return (f'OUT-{os.path.basename(img)}',)


nodes.Qwen3_VQA.inference = interrupting_inference
raised_batch = None
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    try:
        batch.run(**{**kw, 'force': True})
    except BaseException as e:
        raised_batch = e
chk('BatchCache 把中断异常原样抛出', isinstance(raised_batch, _InterruptProcessingException), repr(raised_batch))
chk('中断停在出事那张图，后续不再跑', calls == ['a.png', 'b.JPG'], str(calls))
chk('中断不计入失败', '失败        : 0' in _buf.getvalue())
chk('报告标注被中断', '!! 被中断' in _buf.getvalue())
chk('中断前完成的图已写缓存', len(nodes._read_entries(img)) >= 1)
nodes.Qwen3_VQA.inference = fake_inference

# ================================================================ 4. 接口
print('\n=== 4. 接口 ===')


class _Req:
    def __init__(self, **q):
        self.query = q


async def main():
    n_a = len(nodes._read_entries(img))
    r = await REG[('GET', '/qwen3_vqa/cache/list')](_Req(image_path=img))
    chk('list 返回 ids + entries + next_id',
        len(r.payload['ids']) == n_a == len(r.payload['entries']) and bool(r.payload['next_id']),
        f'(a.png 共 {n_a} 条)')
    chk('list 摘要正确', r.payload['entries'][0]['summary'] == '描述这张图')

    r = await REG[('GET', '/qwen3_vqa/cache/get')](_Req(image_path=img, id=r.payload['ids'][0]))
    chk('get 返回完整条目', 'output' in r.payload['entry'] and 'params' in r.payload['entry'])
    r = await REG[('GET', '/qwen3_vqa/cache/get')](_Req(image_path=img, id='nope'))
    chk('get 未命中 404', r.status == 404)

    r = await REG[('GET', '/qwen3_vqa/batch/scan')](_Req(directory=root, recursive='0'))
    d = r.payload
    chk('scan 统计数量', d['total'] == 3 and d['cached'] == 3 and d['pending'] == 0,
        json.dumps({k: d[k] for k in ('total', 'cached', 'pending')}))
    chk('scan 带每张图的缓存标记', len(d['files']) == 3 and all(f['cached'] for f in d['files']))
    chk('scan 不是写操作', len(nodes._read_entries(img)) == n_a)
    chk('scan 支持递归', (await REG[('GET', '/qwen3_vqa/batch/scan')](_Req(directory=root, recursive='1'))).payload['total'] == 4)
    chk('scan 缺参数 400', (await REG[('GET', '/qwen3_vqa/batch/scan')](_Req())).status == 400)
    chk('scan 非法目录 400', (await REG[('GET', '/qwen3_vqa/batch/scan')](_Req(directory=root + '_nope'))).status == 400)

    r = await REG[('GET', '/qwen3_vqa/cache/list')](_Req(image_path=''))
    chk('list 空参数容错', r.payload == {'ids': [], 'entries': [], 'next_id': None})
    r = await REG[('DELETE', '/qwen3_vqa/cache/delete')](_Req(image_path=img, id='nope'))
    chk('delete 不存在返回 ok=False', r.payload['ok'] is False)


asyncio.run(main())

# ================================================================ 5. 节点注册
print('\n=== 5. 节点注册与注意力选项 ===')
chk('批量节点已注册', hasattr(nodes, 'Qwen3_VL_BatchCache'))
chk('批量节点有 directory 参数', 'directory' in nodes.Qwen3_VL_BatchCache.INPUT_TYPES()['required'])
chk('内置兜底模型列表 9 项', len(nodes.MODEL_CHOICES) == 9)
chk('attention 选项含 sage', 'sage' in nodes.ATTENTION_CHOICES, str(nodes.ATTENTION_CHOICES))
chk('attention 默认值仍是 eager', nodes.ATTENTION_CHOICES[0] == 'eager')
chk('sage 已在 transformers 侧注册成功', nodes._SAGE_REGISTERED is True)
chk('两个节点的 attention 选项一致',
    nodes.Qwen3_VL_BatchCache.INPUT_TYPES()['required']['attention'][0] == nodes.ATTENTION_CHOICES)

# 上游 f1061fe 把 model 下拉从硬编码列表改成"动态扫描 models/prompt_generator"。
# 这里实测 scan_model_choices()：存在目录时只挑名字含 Qwen3-VL 的，不存在时回退内置列表。
print('  -- scan_model_choices（动态模型下拉）--')
_root = tempfile.mkdtemp(prefix='qvqa_models_')
_pg = os.path.join(_root, 'prompt_generator')
os.makedirs(_pg)
for _n in ('Qwen3-VL-4B-Instruct-FP8', 'Qwen3-VL-8B-Thinking', 'some-other-model', 'readme.txt'):
    open(os.path.join(_pg, _n), 'w').close()
_saved_models_dir = nodes.folder_paths.models_dir
try:
    nodes.folder_paths.models_dir = _root
    _scanned = nodes.scan_model_choices()
    chk('动态扫描只挑名字含 Qwen3-VL 的条目',
        sorted(_scanned) == ['Qwen3-VL-4B-Instruct-FP8', 'Qwen3-VL-8B-Thinking'], str(_scanned))
    chk('两个节点的 model 下拉都走同一个扫描函数',
        nodes.Qwen3_VQA.INPUT_TYPES()['required']['model'][0] == _scanned
        and nodes.Qwen3_VL_BatchCache.INPUT_TYPES()['required']['model'][0] == _scanned)
finally:
    nodes.folder_paths.models_dir = _saved_models_dir
    shutil.rmtree(_root, ignore_errors=True)

try:
    nodes.folder_paths.models_dir = os.path.join(tempfile.gettempdir(), 'qvqa_no_such_dir_xyz')
    chk('model 目录不存在 → 回退内置列表（不能让 INPUT_TYPES 抛异常）',
        nodes.scan_model_choices() == nodes.MODEL_CHOICES)
finally:
    nodes.folder_paths.models_dir = _saved_models_dir

# ============================================== 6. 加载参数变化必须触发重载
# ComfyUI 会缓存节点实例（execution.py: caches.objects.get(unique_id)），
# 所以 attention / min_pixels / max_pixels 这些"只在加载时生效"的参数
# 如果不纳入重载判断，改了控件当轮不会生效（曾是一个真 bug）。
print('\n=== 6. 加载参数变化必须触发重载 ===')

# 第 3 节把 inference 换成了假函数，这里换回真的
nodes.Qwen3_VQA.inference = _REAL_INFERENCE

_load_calls = []


class _FakeInputs:
    def __init__(self):
        self.input_ids = [[1, 2, 3]]

    def keys(self):
        return ['input_ids']

    def __getitem__(self, k):
        return getattr(self, k)

    def to(self, device):
        return self


class _FakeProcessor:
    def apply_chat_template(self, messages, **kw):
        return 'PROMPT'

    def __call__(self, **kw):
        return _FakeInputs()

    def batch_decode(self, ids, **kw):
        return ['描述结果']


class _FakeModel:
    def generate(self, **kw):
        return [[1, 2, 3, 9]]


def _fake_proc_loader(path, min_pixels=None, max_pixels=None):
    _load_calls.append(('processor', min_pixels, max_pixels))
    return _FakeProcessor()


def _fake_model_loader(path, **kw):
    _load_calls.append(('model', kw.get('attn_implementation')))
    return _FakeModel()


nodes.AutoProcessor = types.SimpleNamespace(from_pretrained=_fake_proc_loader)
nodes.Qwen3VLForConditionalGeneration = types.SimpleNamespace(from_pretrained=_fake_model_loader)

# 让"模型目录已存在"，否则会走 huggingface 下载
_fp = sys.modules['folder_paths']
_old_models_dir = _fp.models_dir
_models_root = tempfile.mkdtemp(prefix='vqa_models_')
_fp.models_dir = _models_root
os.makedirs(os.path.join(_models_root, 'prompt_generator', 'Qwen3-VL-4B-Instruct-FP8'), exist_ok=True)

_BASE = dict(text='t', model='Qwen3-VL-4B-Instruct-FP8', keep_model_loaded=True,
             temperature=0.7, max_new_tokens=16,
             min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, seed=0,
             quantization='none', use_cache=False, image_path='', prompt_version='<new>',
             image=None, attention='eager')

_runner = nodes.Qwen3_VQA()
_runner.inference(**_BASE)
chk('首次调用加载了 processor + model', len(_load_calls) == 2, f'({_load_calls})')

_n = len(_load_calls)
_runner.inference(**_BASE)
chk('参数不变 → 不重复加载', len(_load_calls) == _n, f'(仍为 {_n})')

_runner.inference(**{**_BASE, 'max_pixels': 4096 * 28 * 28})
chk('改 max_pixels → 触发重载', len(_load_calls) == _n + 2, f'(变成 {len(_load_calls)})')
chk('新 max_pixels 传给了 processor',
    _load_calls[-2] == ('processor', 256 * 28 * 28, 4096 * 28 * 28), f'({_load_calls[-2]})')

_n = len(_load_calls)
_runner.inference(**{**_BASE, 'max_pixels': 4096 * 28 * 28, 'attention': 'sage'})
chk('改 attention → 触发重载', len(_load_calls) == _n + 2)
chk('新 attention 传给了模型', _load_calls[-1] == ('model', 'sage'), f'({_load_calls[-1]})')

_n = len(_load_calls)
_runner.inference(**{**_BASE, 'max_pixels': 4096 * 28 * 28, 'attention': 'sage'})
chk('两项都未变 → 不重复加载', len(_load_calls) == _n, f'(仍为 {_n})')

_n = len(_load_calls)
# 先把新模型的目录建好，否则会去 huggingface 下载
os.makedirs(os.path.join(_models_root, 'prompt_generator', 'Qwen3-VL-8B-Instruct-FP8'),
            exist_ok=True)
_runner.inference(**{**_BASE, 'max_pixels': 4096 * 28 * 28, 'attention': 'sage',
                     'model': 'Qwen3-VL-8B-Instruct-FP8'})
chk('换模型 → 触发重载', len(_load_calls) > _n, f'({len(_load_calls)} > {_n})')

# ================================ 7. attention=sage 缺依赖必须提前报错
# 严格模式：选了 sage 就要用 sage，用不了直接报错（不再静默退回 sdpa）。
# 这一步必须在「下载/加载模型之前」发生，否则用户会白等几分钟才看到错。
# 注意：本节必须跑在第 6 节那套假加载器 + 假 models_dir 还在的时候，
# 否则对照用例会真的去 huggingface 下载模型。
print('\n=== 7. attention=sage 但没装 sageattention → 提前报错 ===')
_saved_sa = sys.modules.get('sageattention')
sys.modules['sageattention'] = None      # 之后 import sageattention 会抛 ImportError
try:
    _n7 = len(_load_calls)
    _bad = nodes.Qwen3_VQA()
    try:
        _bad.inference(**{**_BASE, 'attention': 'sage'})
        chk('没装 sageattention 时 attention=sage → 抛错', False, '(居然没抛)')
    except RuntimeError as e:
        msg = str(e)
        chk('没装 sageattention 时 attention=sage → 抛错 RuntimeError',
            'sageattention' in msg)
        chk('  └ 错误信息带安装指引', 'pip install sageattention' in msg)
        chk('  └ 错误信息给出替代方案', 'sdpa' in msg and 'eager' in msg)
        chk('  └ 加载前就拦住了（没有新增任何加载）', len(_load_calls) == _n7,
            f'(新增 {len(_load_calls) - _n7} 次加载)')
    except Exception as e:
        chk('没装 sageattention 时 attention=sage → 抛错', False,
            f'(抛了 {type(e).__name__}: {e!r})')

    # 对照：attention 不是 sage 时，即使没有 sageattention 也不该被拦
    _n7 = len(_load_calls)
    try:
        nodes.Qwen3_VQA().inference(**{**_BASE, 'attention': 'eager'})
        chk('对照：attention=eager 时不检查 sageattention（照常加载）',
            len(_load_calls) > _n7, f'(新增 {len(_load_calls) - _n7} 次加载)')
    except Exception as e:
        chk('对照：attention=eager 时不检查 sageattention（照常加载）', False,
            f'({type(e).__name__}: {e})')
finally:
    if _saved_sa is None:
        sys.modules.pop('sageattention', None)
    else:
        sys.modules['sageattention'] = _saved_sa

# ================================================================ 7.5 显存让位
print('\n=== 7.5 新 prompt 不含 VQA → 释放模型显存（反向让位） ===')


class _FakeWeight:
    """能挡住 `del` 的哑模型对象，仅用于验证释放逻辑。"""


_inst75 = nodes.Qwen3_VQA()
_inst75.model = _FakeWeight()
_inst75.processor = _FakeWeight()
_r1 = _inst75.release_model('测试')
chk('release_model 有模型时返回 True', _r1 is True)
chk('release_model 后 model/processor 均置空',
    _inst75.model is None and _inst75.processor is None)
chk('release_model 重复调用返回 False', _inst75.release_model() is False)

# 当前实例载着哑模型，看 prompt 队列钩子的判定
_inst75.model = _FakeWeight()
_inst75.processor = _FakeWeight()
_vqa_prompt = {'1': {'class_type': 'Qwen3_VQA'}, '2': {'class_type': 'KSampler'}}
nodes._maybe_release_for_prompt(_vqa_prompt)
chk('prompt 含 Qwen3_VQA → 不释放', _inst75.model is not None)

_other_prompt = {'1': {'class_type': 'KSampler'}, '2': {'class_type': 'VAEDecode'}}
nodes._maybe_release_for_prompt(_other_prompt)
chk('prompt 不含任何 VQA 节点 → 释放', _inst75.model is None)

_non_dict = ['not', 'a', 'prompt']
chk('非 dict prompt → 安全跳过', nodes._maybe_release_for_prompt(_non_dict) == 0)

# 收尾：还原 models_dir 并清掉临时目录
_fp.models_dir = _old_models_dir
shutil.rmtree(_models_root, ignore_errors=True)

shutil.rmtree(root)
print(f"\n结果: {sum(OK)}/{len(OK)} 通过")
sys.exit(0 if all(OK) else 1)

"""Sinh notebook cho lab Day 10 từ một nguồn duy nhất.

Chạy:  python kaggle/build.py

Sinh ra:
  lab_day10.ipynb     bản đầy đủ (E0-E5) — file nộp trong repo
  kaggle/validate/    kernel 1: kiểm tra pipeline + E0
  kaggle/full/        kernel 2: E1, E2, E3, E4
  kaggle/grpo/        kernel 3: E5 (GRPO)

Tách kernel vì mỗi lần chạy GPU trên Kaggle tốn hàng chục phút: một lỗi ở
stage sau không được làm mất kết quả của stage trước.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

USER = "namnguynnnn"
DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
SLUG = {"validate": "k4-day10-validate", "full": "k4-day10-full", "grpo": "k4-day10-grpo"}
TITLE = {
    "validate": "K4 Day10 validate + E0",
    "full": "K4 Day10 E1 E2 E3 E4",
    "grpo": "K4 Day10 E5 GRPO",
}

# --------------------------------------------------------------------------
# Cell. Tag = tập kernel mà cell đó xuất hiện trong đó.
#   validate / full / grpo : kernel tương ứng
#   notebook               : lab_day10.ipynb (bản nộp)
# --------------------------------------------------------------------------

MD_TITLE = """# Lab Day 10 — Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường

Ảnh dashcam 640×360 → model trả một JSON `[{"label": ..., "bbox_2d": [x1,y1,x2,y2]}]`
với toạ độ trên **thang 0–1000**.

Notebook chạy được trên cả **Kaggle** và **Google Colab**:

| | Kaggle | Colab |
|---|---|---|
| Mã nguồn | `git clone` repo | `git clone` repo |
| Dữ liệu | dataset gắn sẵn ở `/kaggle/input/` | Kaggle API qua Colab Secrets |
| GPU | T4 (Accelerator → GPU T4) | T4 |

Đọc `README.md` và `GUIDE.md` trước. `scripts/` và `tests/` là công cụ của đề,
**không được sửa**."""

MD_ENV = "## 0. Môi trường"

CODE_ENV = '''
# Trên Colab/Kaggle thì repo chưa có sẵn -> tự clone.
import os, sys, json, re, time, random, subprocess
from pathlib import Path

IN_KAGGLE = Path('/kaggle/input').is_dir()
IN_COLAB  = (not IN_KAGGLE) and Path('/content').is_dir()

REPO_URL = 'https://github.com/nghon4maeri/K4-Track-4-Day10-VisionLanguageModel.git'
REPO_DIR = 'K4-Track-4-Day10-VisionLanguageModel'

if not Path('scripts/labcommon.py').exists():
    if not Path(REPO_DIR).exists():
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
print('cwd    =', ROOT)
print('kaggle =', IN_KAGGLE, '| colab =', IN_COLAB)
'''

CODE_DEPS = '''
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                '-r', 'requirements-colab.txt'], check=True)

import torch, transformers, peft, accelerate, bitsandbytes
print('torch        ', torch.__version__)
print('transformers ', transformers.__version__)
print('peft         ', peft.__version__)
print('accelerate   ', accelerate.__version__)
print('bitsandbytes ', bitsandbytes.__version__)

assert torch.cuda.is_available(), 'Chưa có GPU: chọn Runtime/Accelerator -> GPU T4.'
cc = torch.cuda.get_device_capability(0)
print('gpu          ', torch.cuda.get_device_name(0), '| cc', cc)
print('vram         ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
# bitsandbytes 4-bit cần compute capability >= 7.0 (T4 = 7.5); P100 (6.0) không chạy được.
assert cc[0] >= 7, f'GPU cc={cc} < 7.0: bitsandbytes 4-bit không hỗ trợ. Hãy chọn T4.'
'''

MD_DATA = "## 1. Dữ liệu: tải và chia split theo phiên quay"

CODE_DATA = '''
# Kaggle: dataset đã gắn làm input -> /kaggle/input/<slug>/
# Colab : lấy khoá Kaggle từ Colab Secrets (KHÔNG ghi khoá vào notebook)
if IN_KAGGLE:
    roots = [p.parent for p in Path('/kaggle/input').rglob('labels-YOLO')
             if (p.parent / 'images').is_dir()]
    assert roots, 'không thấy cặp images/ + labels-YOLO/ trong /kaggle/input'
    CMD = [sys.executable, 'scripts/prepare_data.py', '--source', str(sorted(roots)[0])]
else:
    try:
        from google.colab import userdata
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    except Exception as e:
        print('không đọc được Colab Secrets:', e)
    CMD = [sys.executable, 'scripts/prepare_data.py', '--kaggle']

print('$', ' '.join(CMD))
subprocess.run(CMD, check=True)
'''

CODE_SPLITS = '''
from collections import Counter

meta = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))
DATASET_VERSION = meta['dataset_version']
print('dataset_version:', DATASET_VERSION)

for sp in ('train', 'val', 'test'):
    rows = [json.loads(l) for l in open(f'data/splits/{sp}.jsonl', encoding='utf-8')]
    c = Counter(o['label'] for r in rows for o in r['objects'])
    print(f'{sp:5s} {len(rows):5d} ảnh  boxes={dict(c)}')

# Chia theo phiên quay: không phiên nào được nằm ở hai split (chống rò rỉ
# vì các frame liền kề trong cùng phiên gần như trùng nhau).
sess = {sp: {json.loads(l)['session'] for l in open(f'data/splits/{sp}.jsonl', encoding='utf-8')}
        for sp in ('train', 'val', 'test')}
assert not (sess['train'] & sess['val']), 'rò rỉ session train/val'
assert not (sess['train'] & sess['test']), 'rò rỉ session train/test'
assert not (sess['val'] & sess['test']), 'rò rỉ session val/test'
print('OK: các session tách biệt hoàn toàn giữa 3 split')
'''

MD_LABELS = "## 2. Nhãn: YOLO → JSON thang 0–1000"

CODE_LABELS = '''
CLASSES = ['pothole', 'crack', 'manhole']


def yolo_to_box(label, yolo):
    """Nhãn YOLO (chuẩn hoá [0,1], tâm + kích thước) -> bbox thang 0-1000."""
    xc, yc, w, h = yolo
    x1, y1 = (xc - w / 2) * 1000, (yc - h / 2) * 1000
    x2, y2 = (xc + w / 2) * 1000, (yc + h / 2) * 1000

    def clamp(v):
        return int(min(1000, max(0, round(v))))

    return {'label': label, 'bbox_2d': [clamp(x1), clamp(y1), clamp(x2), clamp(y2)]}


def objects_to_answer(objects):
    """JSON gọn, sắp trên->dưới rồi trái->phải để thứ tự ổn định cho model học."""
    boxes = [yolo_to_box(o['label'], o['yolo']) for o in objects]
    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0]))
    return json.dumps(boxes, separators=(',', ':'))


def objects_to_answer_px(objects, width=640, height=360):
    """Như trên nhưng toạ độ là pixel tuyệt đối (dùng cho ablation coord_scale)."""
    boxes = []
    for o in objects:
        xc, yc, w, h = o['yolo']
        x1, y1 = (xc - w / 2) * width, (yc - h / 2) * height
        x2, y2 = (xc + w / 2) * width, (yc + h / 2) * height

        def clamp(v, mx):
            return int(min(mx, max(0, round(v))))

        boxes.append({'label': o['label'],
                      'bbox_2d': [clamp(x1, width), clamp(y1, height),
                                  clamp(x2, width), clamp(y2, height)]})
    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0]))
    return json.dumps(boxes, separators=(',', ':'))


PROMPT = ('You are a road damage detector. Return a JSON list of objects with keys '
          '"label" and "bbox_2d". Labels are one of: pothole, crack, manhole. '
          'Coordinates are integers on the 0-1000 scale relative to image width/height. '
          'Return [] if none. Output only the JSON, no explanation.')

PROMPT_PIXEL = ('You are a road damage detector. Return a JSON list of objects with keys '
                '"label" and "bbox_2d". Labels are one of: pothole, crack, manhole. '
                'Coordinates are integer pixels in the original 640x360 image. '
                'Return [] if none. Output only the JSON, no explanation.')

_entries_train = [json.loads(l) for l in open('data/splits/train.jsonl', encoding='utf-8')]
print('đáp án 0-1000:', objects_to_answer(_entries_train[0]['objects'])[:130])
print('đáp án pixel :', objects_to_answer_px(_entries_train[0]['objects'],
      _entries_train[0]['width'], _entries_train[0]['height'])[:130])
'''

MD_CHECK_LABELS = "### Kiểm tra bước 2 — quy ngược 0–1000 về pixel rồi so với box gốc"

CODE_CHECK_LABELS = '''
from PIL import Image, ImageDraw

_e = _entries_train[0]
_w, _h = _e['width'], _e['height']
_back = json.loads(objects_to_answer(_e['objects']))          # box qua thang 0-1000
_direct = [{'label': o['label'], 'bbox_2d': [
    int(min(_w, max(0, round((o['yolo'][0] - o['yolo'][2] / 2) * _w)))),
    int(min(_h, max(0, round((o['yolo'][1] - o['yolo'][3] / 2) * _h)))),
    int(min(_w, max(0, round((o['yolo'][0] + o['yolo'][2] / 2) * _w)))),
    int(min(_h, max(0, round((o['yolo'][1] + o['yolo'][3] / 2) * _h))))]}
    for o in _e['objects']]                                    # box tính thẳng ra pixel

_err = 0.0
for b in _back:
    px = [b['bbox_2d'][0] / 1000 * _w, b['bbox_2d'][1] / 1000 * _h,
          b['bbox_2d'][2] / 1000 * _w, b['bbox_2d'][3] / 1000 * _h]
    _err = max(_err, min((sum(abs(px[k] - d['bbox_2d'][k]) for k in range(4))
                          for d in _direct), default=0.0))
print(f'sai số lớn nhất khi quy ngược 0-1000 -> pixel: {_err:.2f} px')
assert _err <= 2.0, 'quy ngược lệch quá 2 px — kiểm tra lại yolo_to_box'

_img = Image.open(Path('data') / _e['image_path']).convert('RGB')
_dr = ImageDraw.Draw(_img)
for b in _back:
    x1, y1, x2, y2 = b['bbox_2d']
    _dr.rectangle([x1 / 1000 * _w, y1 / 1000 * _h, x2 / 1000 * _w, y2 / 1000 * _h],
                  outline=(255, 0, 0), width=3)
    _dr.text((x1 / 1000 * _w + 4, y1 / 1000 * _h + 4), b['label'], fill=(255, 255, 0))
print('số box vẽ được:', len(_back))
_img
'''

MD_INFER = "## 3. Nạp model và các hàm dùng chung"

CODE_INFER = '''
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'

# ViT và lm_head KHÔNG lượng tử hoá -> giữ ViT (kể cả projector) ở fp16.
# Nếu ViT bị lượng tử hoá 4-bit thì trọng số merger là trọng số lượng tử,
# không thể huấn luyện đầy đủ ở E2/E3.
BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=['visual', 'lm_head'],
)


def load_base():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=BNB, dtype=torch.float16, device_map={'': 0})
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = 'left'
    return model, proc


def infer_once(model, proc, image, prompt, max_new_tokens=256):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                             {'type': 'text', 'text': prompt}]}]
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_prompt = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0]


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\\n')


def run_inference(name, adapter_dir=None, projector_path=None, prompt=PROMPT,
                  split='test', out_path=None, limit=None):
    """Nạp model gốc (4-bit) -> adapter LoRA -> tham số projector đã huấn luyện."""
    model, proc = load_base()
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if projector_path is not None and Path(projector_path).exists():
        extra = torch.load(projector_path, map_location='cpu')
        n_loaded = 0
        for n, p in model.named_parameters():
            if n in extra:
                p.data.copy_(extra[n].to(device=p.device, dtype=p.dtype))
                n_loaded += 1
        # Quên nạp lại projector là lỗi hay gặp nhất: E2/E3 sẽ trông y hệt E1.
        assert n_loaded > 0, f'không nạp được tensor nào từ {projector_path}'
        print(f'nạp lại {n_loaded} tensor projector từ {projector_path}')
    model.eval()

    entries = read_jsonl(f'data/splits/{split}.jsonl')
    if limit:
        entries = entries[:limit]
    outs = []
    for i, e in enumerate(entries):
        img = Image.open(Path('data') / e['image_path']).convert('RGB')
        outs.append({'image_id': e['image_id'], 'raw': infer_once(model, proc, img, prompt)})
        if i % 50 == 0:
            print(name, i, outs[-1]['raw'][:60])
    out_path = out_path or f'work/{name.lower()}.jsonl'
    write_jsonl(out_path, outs)
    print('ghi', out_path, len(outs), 'dòng')
    del model
    torch.cuda.empty_cache()
    return out_path


def run_score(name, pred, out, coord='norm1000', split='test', extra_meta=None):
    cmd = [sys.executable, 'scripts/score.py', '--pred', pred, '--name', name,
           '--out', out, '--split', split, '--coord', coord]
    for k, v in (extra_meta or {}).items():
        cmd += ['--meta', f'{k}={v}']
    print('$', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def run_meta(name):
    p = Path(f'work/{name}_run.json')
    return json.loads(p.read_text()) if p.exists() else {}
'''

MD_E0 = "## 4. E0: zero-shot"

CODE_E0 = '''
model, proc = load_base()
_e0_entries = read_jsonl('data/splits/test.jsonl')
_e0_outs = []
for i, e in enumerate(_e0_entries):
    img = Image.open(Path('data') / e['image_path']).convert('RGB')
    _e0_outs.append({'image_id': e['image_id'], 'raw': infer_once(model, proc, img, PROMPT)})
    if i % 50 == 0:
        print('E0', i, _e0_outs[-1]['raw'][:60])
write_jsonl('work/e0.jsonl', _e0_outs)
print('E0 xong:', len(_e0_outs), 'ảnh')
'''

CODE_E0_SCORE = '''
run_score('E0', 'work/e0.jsonl', 'report/evidence/e0.json')

# Xem output thô + tỉ lệ parse hợp lệ để trả lời câu hỏi 2 của REPORT.md.
_rec0 = json.loads(Path('report/evidence/e0.json').read_text(encoding='utf-8'))
print('parse_valid_rate =', _rec0['parse_valid_rate'])
print('F1 =', round(_rec0['micro']['f1'], 4), '| FP/ảnh =', round(_rec0['fp_per_image'], 2))
for r in _e0_outs[:5]:
    print('  raw:', repr(r['raw'][:170]))
'''

MD_ENCODE = "## 5. Dựng dữ liệu huấn luyện: chat template và mask loss"

CODE_ENCODE = '''
def encode_example(image, prompt, answer, processor):
    """Trả về full_enc kèm labels; chỉ phần trả lời được tính loss."""
    user_msg = {'role': 'user', 'content': [{'type': 'image', 'image': image},
                                            {'type': 'text', 'text': prompt}]}
    asst_msg = {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}

    prompt_enc = processor.apply_chat_template(
        [user_msg], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt')
    full_enc = processor.apply_chat_template(
        [user_msg, asst_msg], tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt')

    n = prompt_enc['input_ids'].shape[1]
    # prompt phải là tiền tố của chuỗi đầy đủ, nếu không mask sẽ lệch.
    assert torch.equal(full_enc['input_ids'][0, :n], prompt_enc['input_ids'][0]), \\
        'prompt không phải tiền tố của chuỗi đầy đủ'

    labels = full_enc['input_ids'].clone()
    labels[:, :n] = -100          # không tính loss trên prompt và token ảnh
    out = dict(full_enc)
    out['labels'] = labels
    return out, prompt_enc


def collate(batch):
    """Pad các trường chuỗi; pixel_values / image_grid_thw thì nối, KHÔNG pad."""
    out = {}
    for k in ['input_ids', 'attention_mask', 'mm_token_type_ids', 'labels']:
        if k not in batch[0]:
            continue
        maxlen = max(b[k].shape[-1] for b in batch)
        pad_val = -100 if k == 'labels' else 0
        tensors = []
        for b in batch:
            t = b[k]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            tensors.append(torch.nn.functional.pad(t, (0, maxlen - t.shape[-1]), value=pad_val))
        out[k] = torch.cat(tensors, dim=0)
    for k in ['pixel_values', 'image_grid_thw']:
        if k in batch[0]:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out
'''

MD_SETUP = "## 6. Setup LoRA / đóng băng cho E1, E2, E3"

CODE_SETUP = '''
from peft import LoraConfig, get_peft_model

LLM_RE = r'language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)'
VIT_RE = r'visual\\.blocks\\.\\d+\\.(attn\\.(qkv|proj)|mlp\\.(linear_fc1|linear_fc2))'
PROJ_KEY = ('visual.merger', 'visual.deepstack_merger_list')


def apply_setup(model, setup, rank=16, alpha=32):
    """Đóng băng tất cả, rồi mở LoRA (LLM, và ViT nếu E3) + projector (nếu E2/E3)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()          # cần cho gradient checkpointing

    patterns = [LLM_RE]
    if setup == 'E3':
        patterns.append(VIT_RE)                 # LoRA thêm trên attention/MLP của ViT
    combined = '|'.join(f'({p})' for p in patterns)

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                     target_modules=combined, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)

    if setup in ('E2', 'E3'):
        # Projector được huấn luyện ĐẦY ĐỦ (không phải LoRA) và giữ ở float32.
        for n, p in model.named_parameters():
            if any(k in n for k in PROJ_KEY):
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)

    matched = [n for n, _ in model.named_modules() if re.search(combined, n)]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{setup}: matched_modules={len(matched)}  trainable_params={n_train:,}')
    return model
'''

MD_VALIDATE = "## 7. Kiểm tra pipeline trước khi huấn luyện"

CODE_VALIDATE = '''
# Gom các kiểm tra của GUIDE lại. Mỗi khối bọc try/except để một lỗi bất ngờ
# vẫn ghi lại được và không chặn các kiểm tra còn lại — mỗi lần chạy GPU chỉ có
# một cơ hội để lấy thông tin.
import traceback

CHECKS = []


def check(ok, msg):
    CHECKS.append((bool(ok), msg))
    print(('  OK   ' if ok else '  FAIL ') + msg)


def guard(label, fn):
    print(f'--- {label} ---')
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        CHECKS.append((False, f'{label}: {type(exc).__name__}: {exc}'))


def _check_encode():
    e = _entries_train[0]
    img = Image.open(Path('data') / e['image_path']).convert('RGB')
    ans = objects_to_answer(e['objects'])
    enc, pe = encode_example(img, PROMPT, ans, proc)

    n_prompt = pe['input_ids'].shape[1]
    n_full = enc['input_ids'].shape[1]
    mm = enc.get('mm_token_type_ids')
    n_img_tok = int((mm == 1).sum()) if mm is not None else -1
    decoded = proc.tokenizer.decode(enc['input_ids'][0][enc['labels'][0] != -100],
                                    skip_special_tokens=False)

    print('độ dài chuỗi đầy đủ :', n_full)
    print('độ dài prompt       :', n_prompt)
    print('token trả lời       :', n_full - n_prompt)
    print('token ảnh           :', n_img_tok, '(GUIDE nói 220 cho ảnh 640x360)')
    print('giải mã phần labels :', repr(decoded[:130]))

    check(n_img_tok == 220, f'số token ảnh = {n_img_tok} (mong đợi 220)')
    check(torch.equal(enc['input_ids'][0, :n_prompt], pe['input_ids'][0]),
          'prompt là tiền tố của chuỗi đầy đủ')
    check(all(int(x) == -100 for x in enc['labels'][0][:n_prompt]),
          'toàn bộ token prompt và token ảnh đã bị mask (-100)')
    check(decoded.startswith(ans), 'phần học giải mã ra đúng JSON đáp án')
    check('<|im_end|>' in decoded, 'có token kết thúc <|im_end|> trong nhãn')

    b = [encode_example(Image.open(Path('data') / x['image_path']).convert('RGB'),
                        PROMPT, objects_to_answer(x['objects']), proc)[0]
         for x in _entries_train[:2]]
    c = collate(b)
    check(c['input_ids'].shape[0] == 2 and c['labels'].shape == c['input_ids'].shape,
          f"collate cho shape {tuple(c['input_ids'].shape)}")
    check(c['pixel_values'].shape[0] == 2, 'collate nối pixel_values theo chiều 0')


def _check_vit_dtype():
    merger = dict(model.named_parameters())['visual.merger.mlp.0.weight']
    print('dtype visual.merger:', merger.dtype)
    check(merger.dtype == torch.float16,
          'visual.merger ở fp16 (ViT không bị lượng tử hoá 4-bit)')


guard('Bước 4: encode_example + collate', _check_encode)
print()
guard('Bước 3: ViT có bị lượng tử hoá không', _check_vit_dtype)
'''

CODE_SETUP_CHECK = '''
_STATE = {}


def _check_targets():
    names = [n for n, _ in model.named_modules()]
    n_llm = len([n for n in names if re.search(LLM_RE, n)])
    n_vit = len([n for n in names if re.search(VIT_RE, n)])
    print(f'LLM={n_llm} ViT={n_vit}')
    check(n_llm == 252, f'LLM có 252 module đích (36x7), đếm được {n_llm}')
    check(n_vit == 96, f'ViT có 96 module đích (24x4), đếm được {n_vit}')


def _check_e2():
    peft_e2 = apply_setup(model, 'E2')                 # E2 = E1 + mở projector (~109M)
    _STATE['n_e2'] = sum(p.numel() for p in peft_e2.parameters() if p.requires_grad)
    proj = [(n, p) for n, p in peft_e2.named_parameters() if any(k in n for k in PROJ_KEY)]
    _STATE['n_proj'] = sum(p.numel() for _, p in proj)
    print(f"E2 trainable={_STATE['n_e2']:,} | riêng projector={_STATE['n_proj']:,}")
    check(len(proj) > 0, f'E2 mở {len(proj)} tensor projector')
    check(all(p.dtype == torch.float32 for _, p in proj), 'projector ở float32')
    check(all(p.requires_grad for _, p in proj), 'projector được mở huấn luyện')
    del peft_e2


def _check_e3():
    global model
    del model
    torch.cuda.empty_cache()
    model_e3, _ = load_base()                          # E3 = E2 + LoRA trên ViT
    peft_e3 = apply_setup(model_e3, 'E3')
    matched = [n for n, _ in peft_e3.named_modules()
               if re.search(f'({LLM_RE})|({VIT_RE})', n)]
    n_e3 = sum(p.numel() for p in peft_e3.parameters() if p.requires_grad)
    print(f'E3 matched={len(matched)} trainable={n_e3:,}')
    check(len(matched) == 348, f'E3 có 348 module đích, đếm được {len(matched)}')
    if 'n_e2' in _STATE:
        check(n_e3 > _STATE['n_e2'],
              f"E3 trainable={n_e3:,} > E2 trainable={_STATE['n_e2']:,} "
              f"(LoRA trên ViT thêm {n_e3 - _STATE['n_e2']:,})")
    del model_e3, peft_e3
    torch.cuda.empty_cache()


guard('Bước 5: module đích của LoRA', _check_targets)
guard('Bước 5: E2 mở projector', _check_e2)
guard('Bước 5: E3 thêm LoRA trên ViT', _check_e3)
'''

CODE_SMOKE = '''
# Huấn luyện 2 bước để chắc vòng train + lưu/nạp projector hoạt động,
# rồi suy luận thử 3 ảnh qua đúng đường đi của E2 (base -> adapter -> projector).
def _check_smoke():
    a_smoke, p_smoke, r_smoke = train_run('E2', 'SMOKE', max_steps=2, save_every=0)
    check(r_smoke['trainable_params'] > 0,
          f"train_run chạy được, trainable={r_smoke['trainable_params']:,}")
    check(r_smoke['loss_last'] == r_smoke['loss_last'], 'loss hữu hạn (không NaN)')
    check(Path(p_smoke).exists(), f'projector được lưu riêng tại {p_smoke}')

    run_inference('SMOKE', adapter_dir=a_smoke, projector_path=p_smoke,
                  split='val', out_path='work/smoke.jsonl', limit=3)
    check(Path('work/smoke.jsonl').exists(), 'suy luận với adapter + projector chạy được')


guard('Bước 6: vòng huấn luyện + lưu/nạp projector', _check_smoke)
'''

CODE_VALIDATE_SUMMARY = '''
print('=' * 72)
_bad = [m for ok, m in CHECKS if not ok]
print(f'{len(CHECKS) - len(_bad)}/{len(CHECKS)} kiểm tra ĐẠT')
for m in _bad:
    print('  FAIL:', m)
print('=' * 72)
if _bad:
    raise AssertionError(f'{len(_bad)} kiểm tra thất bại — xem danh sách ở trên')
'''

MD_TRAIN = "## 8. Vòng huấn luyện"

CODE_TRAIN = '''
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

LR = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 300


class VLDataset(Dataset):
    def __init__(self, entries, prompt, answer_fn, processor, fraction=1.0, seed=42):
        rng = random.Random(seed)
        idx = list(range(len(entries)))
        rng.shuffle(idx)
        k = max(1, int(len(entries) * fraction)) if fraction < 1 else len(entries)
        self.sel = [entries[i] for i in idx[:k]]
        self.prompt, self.answer_fn, self.processor = prompt, answer_fn, processor

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        e = self.sel[i]
        img = Image.open(Path('data') / e['image_path']).convert('RGB')
        enc, _ = encode_example(img, self.prompt, self.answer_fn(e), self.processor)
        return enc


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def save_ckpt(model, path):
    """save_pretrained của peft CHỈ lưu adapter LoRA; projector phải lưu riêng."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    extra = {n: p.detach().cpu() for n, p in model.named_parameters()
             if p.requires_grad and any(k in n for k in PROJ_KEY)}
    if extra:
        torch.save(extra, path / 'projector.pt')
        print(f'  lưu {len(extra)} tensor projector -> {path}/projector.pt')
    return path


def train_run(setup, name, rank=16, alpha=32, train_fraction=1.0,
              coord_scale='norm1000', max_steps=MAX_STEPS, save_every=100):
    prompt = PROMPT if coord_scale == 'norm1000' else PROMPT_PIXEL
    if coord_scale == 'norm1000':
        def answer_fn(e):
            return objects_to_answer(e['objects'])
    else:
        def answer_fn(e):
            return objects_to_answer_px(e['objects'], e['width'], e['height'])

    model, processor = load_base()
    model = apply_setup(model, setup, rank=rank, alpha=alpha)

    entries = read_jsonl('data/splits/train.jsonl')
    ds = VLDataset(entries, prompt, answer_fn, processor, fraction=train_fraction, seed=42)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    print(f'{name}: {len(ds)} ảnh train | {max_steps} bước tối ưu | '
          f'batch={BATCH_SIZE} x accum={GRAD_ACCUM} | coord={coord_scale}')

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    opt = AdamW(trainable, lr=LR, weight_decay=0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * max_steps)), max_steps)
    scaler = GradScaler('cuda')

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()

    torch.cuda.reset_peak_memory_stats()
    it = _cycle(loader)
    losses, t0 = [], time.time()

    opt.zero_grad()
    for step in range(1, max_steps + 1):
        for _ in range(GRAD_ACCUM):
            batch = next(it)
            batch = {k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**batch).loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError(f'loss không hữu hạn ở bước {step} — nguy cơ tràn số fp16')
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * GRAD_ACCUM)

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        if step % 10 == 0 or step == 1:
            _tail = losses[-GRAD_ACCUM * 10:]
            print(f'  {name} step {step:3d}/{max_steps}  loss {sum(_tail) / len(_tail):.4f}')
        if save_every and step % save_every == 0:
            save_ckpt(model, f'work/{name}_step{step}')

    adapter_dir = save_ckpt(model, f'work/{name}_adapter')
    run = {
        'setup': setup, 'rank': rank, 'alpha': alpha,
        'train_fraction': train_fraction, 'coord_scale': coord_scale,
        'max_steps': max_steps,
        'trainable_params': int(trainable_params),
        'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
        'train_minutes': round((time.time() - t0) / 60, 2),
        'loss_first': round(sum(losses[:20]) / max(1, len(losses[:20])), 4),
        'loss_last': round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
        'losses': [round(v, 4) for v in losses[::max(1, len(losses) // 200)]],
    }
    Path(f'work/{name}_run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(name, {k: v for k, v in run.items() if k != 'losses'})

    del model, opt, scaler
    torch.cuda.empty_cache()
    return adapter_dir, adapter_dir / 'projector.pt', run
'''

MD_BEST = "## 9. Chọn setup tốt nhất trong E1–E3"

CODE_BEST = '''
# Dùng cho E4 (giữ mọi thứ khác) và E5 (khởi tạo từ adapter tốt nhất).
_ev = {}
for _n in ('e1', 'e2', 'e3'):
    _p = Path(f'report/evidence/{_n}.json')
    if _p.exists():
        _ev[_n.upper()] = json.loads(_p.read_text(encoding='utf-8'))
BEST = max(_ev, key=lambda k: _ev[k]['micro']['f1']) if _ev else 'E2'
print('F1:', {k: round(v['micro']['f1'], 4) for k, v in _ev.items()})
print('BEST =', BEST)
'''

MD_E1 = "## 10. E1 — LoRA chỉ trên LLM"

CODE_E1 = '''
a1, p1, r1 = train_run('E1', 'E1')
run_inference('E1', adapter_dir=a1, projector_path=None, out_path='work/e1.jsonl')
run_score('E1', 'work/e1.jsonl', 'report/evidence/e1.json', extra_meta=r1)
'''

MD_E2 = "## 11. E2 — E1 + mở đầy đủ projector"

CODE_E2 = '''
a2, p2, r2 = train_run('E2', 'E2')
run_inference('E2', adapter_dir=a2, projector_path=p2, out_path='work/e2.jsonl')
run_score('E2', 'work/e2.jsonl', 'report/evidence/e2.json', extra_meta=r2)
'''

MD_E3 = "## 12. E3 — E2 + LoRA trên các khối của ViT"

CODE_E3 = '''
a3, p3, r3 = train_run('E3', 'E3')
run_inference('E3', adapter_dir=a3, projector_path=p3, out_path='work/e3.jsonl')
run_score('E3', 'work/e3.jsonl', 'report/evidence/e3.json', extra_meta=r3)
'''

MD_E4 = "## 13. E4 — ablation: thang toạ độ (norm1000 vs pixel)"

CODE_E4 = '''
# E4 đổi ĐÚNG MỘT biến so với setup tốt nhất: thang toạ độ.
# Nhánh norm1000 CHÍNH LÀ setup gốc đã chạy ở trên (GUIDE cho phép dùng lại),
# nhánh pixel phải huấn luyện lại từ đầu với nhãn + prompt ở thang pixel.
a_px, p_px, r_px = train_run(BEST, 'E4_coord_pixel', coord_scale='pixel')
run_inference('E4_coord_pixel', adapter_dir=a_px, projector_path=p_px,
              prompt=PROMPT_PIXEL, out_path='work/e4_coord_pixel.jsonl')
'''

CODE_E4_SCORE = '''
run_score('E4_coord_norm1000', f'work/{BEST.lower()}.jsonl',
          'report/evidence/e4_coord_norm1000.json',
          coord='norm1000', extra_meta=run_meta(BEST))
run_score('E4_coord_pixel', 'work/e4_coord_pixel.jsonl',
          'report/evidence/e4_coord_pixel.json',
          coord='pixel', extra_meta=r_px)
'''

MD_FIG = "## 14. Ảnh minh hoạ dự đoán đúng/sai (cho REPORT.md)"

CODE_FIG = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scripts.score import parse_output

FIGDIR = Path('report/figures')
FIGDIR.mkdir(parents=True, exist_ok=True)


def gt_px(entry):
    out = []
    for o in entry['objects']:
        xc, yc, w, h = o['yolo']
        out.append((o['label'], (xc - w / 2) * 1000, (yc - h / 2) * 1000,
                    (xc + w / 2) * 1000, (yc + h / 2) * 1000))
    return out


def draw(entry, raw, title, path):
    img = Image.open(Path('data') / entry['image_path']).convert('RGB')
    W, H = entry['width'], entry['height']
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(img)
    for lab, x1, y1, x2, y2 in gt_px(entry):            # box đúng: xanh lá
        ax.add_patch(Rectangle((x1 / 1000 * W, y1 / 1000 * H),
                               (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H,
                               fill=False, edgecolor='lime', lw=2))
        ax.text(x1 / 1000 * W, y1 / 1000 * H - 3, f'GT {lab}', color='lime', fontsize=8)
    parsed = parse_output(raw) if raw is not None else None
    if parsed is None:
        ax.text(6, 20, 'JSON hỏng', color='red', fontsize=12)
    else:
        for p in parsed:                                # box dự đoán: đỏ
            x1, y1, x2, y2 = p['bbox']
            ax.add_patch(Rectangle((x1 / 1000 * W, y1 / 1000 * H),
                                   (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H,
                                   fill=False, edgecolor='red', lw=2, ls='--'))
            ax.text(x2 / 1000 * W - 40, y2 / 1000 * H + 10, f"P {p['label']}",
                    color='red', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


_by_id = {e['image_id']: e for e in read_jsonl('data/splits/test.jsonl')}
_figs = []
for _tag in ('e0', 'e1', 'e2', 'e3', 'e4_coord_pixel'):
    _p = Path(f'work/{_tag}.jsonl')
    if not _p.exists():
        continue
    for _row in read_jsonl(_p)[:3]:
        _e = _by_id.get(_row['image_id'])
        if _e is None:
            continue
        _f = FIGDIR / f'{_tag}_{_row["image_id"]}.png'
        _figs.append(str(draw(_e, _row['raw'], f"{_tag} — {_row['image_id']}", _f)))
print(f'đã lưu {len(_figs)} ảnh minh hoạ vào {FIGDIR}')
for _f in _figs[:10]:
    print(' ', _f)
'''

MD_E5 = "## 15. E5 (tuỳ chọn) — GRPO với reward IoU"

CODE_E5_REWARD = '''
from scripts.score import parse_output, iou_xyxy

IOU_THR = 0.5


def reward_iou(completions, gt_boxes=None, **kwargs):
    """0.2 * JSON hợp lệ + 0.8 * (tổng IoU các cặp ghép được) / max(n_gt, n_pred, 1).

    Mẫu số phạt box bịa: thêm box không khớp làm giảm điểm.
    Không có gt mà model cũng trả rỗng thì thưởng 1.0.
    """
    rewards = []
    for comp, gts in zip(completions, gt_boxes or [None] * len(completions)):
        text = comp if isinstance(comp, str) else str(comp)
        pred = parse_output(text)
        if pred is None:                      # JSON hỏng
            rewards.append(0.0)
            continue
        gts = gts or []
        if not gts and not pred:
            rewards.append(1.0)
            continue
        pairs = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(pred):
                if g['label'] == p['label']:
                    v = iou_xyxy(g['bbox'], p['bbox'])
                    if v >= IOU_THR:
                        pairs.append((v, gi, pi))
        pairs.sort(reverse=True)              # ghép tham lam theo IoU giảm dần
        used_g, used_p, total = set(), set(), 0.0
        for v, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            total += v
        rewards.append(0.2 + 0.8 * total / max(len(gts), len(pred), 1))
    return rewards
'''

CODE_E5_TEST = '''
# Reward là Python thuần -> test được không cần GPU (GUIDE bước 9).
def _gt(lab, box):
    return {'label': lab, 'bbox': box}


_g1 = [_gt('pothole', (0, 0, 100, 100))]
_cases = [
    ('hoàn hảo',        '[{"label":"pothole","bbox_2d":[0,0,100,100]}]', _g1, 1.0),
    ('JSON hỏng',       'not json at all',                               _g1, 0.0),
    ('sai lớp',         '[{"label":"crack","bbox_2d":[0,0,100,100]}]',   _g1, 0.2),
    ('thừa box',        '[{"label":"pothole","bbox_2d":[0,0,100,100]},'
                        ' {"label":"pothole","bbox_2d":[500,500,600,600]}]', _g1, 0.6),
    ('rỗng, có gt',     '[]',                                            _g1, 0.2),
    ('rỗng, không gt',  '[]',                                            [],  1.0),
    ('IoU dưới ngưỡng', '[{"label":"pothole","bbox_2d":[80,80,180,180]}]', _g1, 0.2),
]
for _name, _comp, _g, _want in _cases:
    _got = reward_iou([_comp], gt_boxes=[_g])[0]
    assert abs(_got - _want) < 1e-6, f'{_name}: mong {_want}, được {_got}'
    print(f'  OK  {_name:18s} reward={_got:.2f}')

# Bọc trong ```json cũng phải parse được (parse_output tự gỡ fence)
assert reward_iou(['```json\\n[{"label":"pothole","bbox_2d":[0,0,100,100]}]\\n```'],
                  gt_boxes=[_g1])[0] == 1.0
print('  OK  output bọc trong ```json')
print('reward_iou: tất cả test đạt')
'''

CODE_E5_DATASET = '''
from datasets import Dataset


def grpo_dataset(split='train', limit=None):
    """prompt là hội thoại có ảnh; gt_boxes đi kèm để hàm reward dùng."""
    entries = read_jsonl(f'data/splits/{split}.jsonl')
    if limit:
        entries = entries[:limit]
    rows = []
    for e in entries:
        img = Image.open(Path('data') / e['image_path']).convert('RGB')
        gt = [{'label': b['label'], 'bbox': tuple(b['bbox_2d'])}
              for b in json.loads(objects_to_answer(e['objects']))]
        rows.append({
            'prompt': [{'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': PROMPT}]}],
            'gt_boxes': gt,
        })
    return Dataset.from_list(rows)


_ds_smoke = grpo_dataset(limit=2)
print(_ds_smoke)
print('gt ví dụ:', _ds_smoke[0]['gt_boxes'])
'''

CODE_E5_TRAIN = '''
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# Khởi tạo từ adapter tốt nhất của E1-E3 (BEST chọn ở cell trên).
# Khi chạy thành kernel riêng, adapter nằm trong output của kernel `full`
# (gắn qua kernel_sources -> /kaggle/input/k4-day10-full/...).
if Path(f'work/{BEST}_adapter').exists():
    ADAPTER_DIR = Path(f'work/{BEST}_adapter')
else:
    _hits = sorted(Path('/kaggle/input').rglob(f'{BEST}_adapter'))
    assert _hits, f'không tìm thấy adapter {BEST} trong work/ lẫn /kaggle/input'
    ADAPTER_DIR = _hits[0]
PROJECTOR_PATH = ADAPTER_DIR / 'projector.pt'
if not PROJECTOR_PATH.exists():
    PROJECTOR_PATH = None
print('ADAPTER_DIR    =', ADAPTER_DIR)
print('PROJECTOR_PATH =', PROJECTOR_PATH)

model, processor = load_base()
model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
if PROJECTOR_PATH is not None:
    _extra = torch.load(PROJECTOR_PATH, map_location='cpu')
    _n = 0
    for _n_name, _p in model.named_parameters():
        if _n_name in _extra:
            _p.data.copy_(_extra[_n_name].to(device=_p.device, dtype=_p.dtype))
            _n += 1
    print(f'nạp lại {_n} tensor projector')
model.config.use_cache = False

cfg = GRPOConfig(
    output_dir='work/E5_grpo',
    learning_rate=5e-6,
    beta=0.0,                      # không dùng reference model -> tiết kiệm VRAM
    num_generations=4,
    max_completion_length=128,
    max_steps=60,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_steps=30,
    use_vllm=False,
    log_completions=True,          # cần log để phân tích reward hacking
    num_completions_to_print=4,
    temperature=1.0,
    report_to=[],
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_iou,
    args=cfg,
    train_dataset=grpo_dataset(split='train'),
    processing_class=processor,
)
trainer.train()
trainer.save_model('work/E5_grpo_final')
print('GRPO xong')
'''

CODE_E5_EVAL = '''
import pandas as pd

# Log completion để soi reward hacking: model có trả [] hoặc một box khổng lồ không?
_tbl = pd.DataFrame(trainer.state.log_history)
_cols = [c for c in ('step', 'reward', 'reward_std', 'completions/mean_length', 'kl', 'loss')
         if c in _tbl.columns]
print(_tbl[_cols].dropna(how='all').to_string(index=False))
Path('report/evidence').mkdir(parents=True, exist_ok=True)
_tbl.to_csv('report/evidence/e5_grpo_log.csv', index=False)

_log = []
for _step_rows in getattr(trainer, '_logs', {}).get('completions', [])[-8:]:
    for _c in _step_rows:
        _log.append(str(_c))
Path('report/evidence/e5_grpo_completions.txt').write_text(
    '\\n---\\n'.join(_log), encoding='utf-8')
print('số completion lưu lại:', len(_log))
for _c in _log[:5]:
    print(' ', repr(_c[:170]))

# Chấm F1 trên test bằng adapter sau GRPO để so với SFT
run_inference('E5', adapter_dir='work/E5_grpo_final', projector_path=PROJECTOR_PATH,
              out_path='work/e5.jsonl')
run_score('E5', 'work/e5.jsonl', 'report/evidence/e5.json')
'''

# --------------------------------------------------------------------------
# Lắp ghép
# --------------------------------------------------------------------------
ALL = {'validate', 'full', 'grpo', 'notebook'}
BASE = ALL

CELLS = [
    ('md', MD_TITLE, ALL),
    ('md', MD_ENV, BASE),
    ('code', CODE_ENV, BASE),
    ('code', CODE_DEPS, BASE),
    ('md', MD_DATA, BASE),
    ('code', CODE_DATA, BASE),
    ('code', CODE_SPLITS, BASE),
    ('md', MD_LABELS, BASE),
    ('code', CODE_LABELS, BASE),
    ('md', MD_CHECK_LABELS, {'validate', 'notebook'}),
    ('code', CODE_CHECK_LABELS, {'validate', 'notebook'}),
    ('md', MD_INFER, BASE),
    ('code', CODE_INFER, BASE),
    ('md', MD_E0, {'validate', 'notebook'}),
    ('code', CODE_E0, {'validate', 'notebook'}),
    ('code', CODE_E0_SCORE, {'validate', 'notebook'}),
    ('md', MD_ENCODE, BASE),
    ('code', CODE_ENCODE, BASE),
    ('md', MD_VALIDATE, {'validate', 'notebook'}),
    ('code', CODE_VALIDATE, {'validate', 'notebook'}),
    ('md', MD_SETUP, BASE),
    ('code', CODE_SETUP, BASE),
    ('code', CODE_SETUP_CHECK, {'validate', 'notebook'}),
    ('md', MD_TRAIN, BASE),
    ('code', CODE_TRAIN, BASE),
    # smoke test dùng train_run() nên phải đứng SAU cell định nghĩa nó
    ('code', CODE_SMOKE, {'validate'}),
    ('code', CODE_VALIDATE_SUMMARY, {'validate'}),
    ('md', MD_BEST, {'full', 'grpo', 'notebook'}),
    ('code', CODE_BEST, {'full', 'grpo', 'notebook'}),
    ('md', MD_E1, {'full', 'notebook'}),
    ('code', CODE_E1, {'full', 'notebook'}),
    ('md', MD_E2, {'full', 'notebook'}),
    ('code', CODE_E2, {'full', 'notebook'}),
    ('md', MD_E3, {'full', 'notebook'}),
    ('code', CODE_E3, {'full', 'notebook'}),
    ('md', MD_E4, {'full', 'notebook'}),
    ('code', CODE_E4, {'full', 'notebook'}),
    ('code', CODE_E4_SCORE, {'full', 'notebook'}),
    ('md', MD_FIG, {'full', 'notebook'}),
    ('code', CODE_FIG, {'full', 'notebook'}),
    ('md', MD_E5, {'grpo', 'notebook'}),
    ('code', CODE_E5_REWARD, {'grpo', 'notebook'}),
    ('code', CODE_E5_TEST, {'grpo', 'notebook'}),
    ('code', CODE_E5_DATASET, {'grpo', 'notebook'}),
    ('code', CODE_E5_TRAIN, {'grpo', 'notebook'}),
    ('code', CODE_E5_EVAL, {'grpo', 'notebook'}),
]

STAGES = ['validate', 'full', 'grpo']


def to_source(text):
    """nbformat: mỗi dòng là một phần tử, kết thúc bằng \\n (trừ dòng cuối)."""
    lines = text.strip('\n').split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def build_notebook(stage):
    cells = []
    for kind, text, tags in CELLS:
        if stage not in tags:
            continue
        kind = 'markdown' if kind == 'md' else kind
        cell = {'cell_type': kind, 'metadata': {}, 'source': to_source(text)}
        if kind == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        cells.append(cell)
    return {
        'cells': cells,
        'metadata': {
            'accelerator': 'GPU',
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def write_metadata(stage):
    return {
        'id': f'{USER}/{SLUG[stage]}',
        'title': TITLE[stage],
        'code_file': f'{SLUG[stage]}.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': 'true',
        'enable_gpu': 'true',
        'enable_tpu': 'false',
        'enable_internet': 'true',
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [DATASET],
        'competition_sources': [],
        'kernel_sources': [f'{USER}/{SLUG["full"]}'] if stage == 'grpo' else [],
        'model_sources': [],
    }


def write(path, nb):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {path.relative_to(ROOT)}  ({len(nb["cells"])} cells)')


def main():
    write(ROOT / 'lab_day10.ipynb', build_notebook('notebook'))
    for stage in STAGES:
        out = HERE / stage
        write(out / f'{SLUG[stage]}.ipynb', build_notebook(stage))
        (out / 'kernel-metadata.json').write_text(
            json.dumps(write_metadata(stage), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

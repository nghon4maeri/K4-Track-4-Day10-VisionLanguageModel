"""Kiểm tra notebook sinh ra: newline trong `source`, cú pháp Python, mục markdown.

Chạy:  python kaggle/check_notebooks.py
"""
import ast
import builtins
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {
    "lab_day10.ipynb": ROOT / "lab_day10.ipynb",
    "validate": ROOT / "kaggle/validate/k4-day10-validate-e0.ipynb",
    "full": ROOT / "kaggle/full/k4-day10-full.ipynb",
    "grpo": ROOT / "kaggle/grpo/k4-day10-grpo.ipynb",
}


class _ModuleScope(ast.NodeVisitor):
    """Thu tên ở CẤP MODULE của một cell.

    Không đi vào thân hàm/lớp: biến cục bộ trong đó chỉ tồn tại lúc hàm chạy,
    không phải lúc cell thực thi.
    """

    def __init__(self):
        self.bound = set()
        self.loaded = []

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)          # không thăm thân hàm

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        pass                               # biến trong lambda là cục bộ

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        else:
            self.loaded.append(node.id)

    def visit_ExceptHandler(self, node):
        # `except X as e` — ast lưu tên này dưới dạng chuỗi, không phải ast.Name
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def _import(self, node):
        for a in node.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    visit_Import = _import
    visit_ImportFrom = _import


def check_cell_order(nb):
    """Bắt lỗi 'dùng tên trước khi cell định nghĩa nó' — cell chạy tuần tự."""
    trees, defined = [], []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            trees.append(None)
            defined.append(set())
            continue
        t = ast.parse("".join(c["source"]))
        v = _ModuleScope()
        v.visit(t)
        trees.append((t, v.loaded))
        defined.append(v.bound)

    later = set().union(*defined) if defined else set()
    bi = set(dir(builtins))
    errs, seen = [], set()
    for i, entry in enumerate(trees):
        if entry is None:
            continue
        _t, loaded = entry
        for nm in loaded:
            # tên gán trong chính cell này coi như đã có (không xét thứ tự dòng
            # trong một cell); chỉ bắt trường hợp tên chỉ được định nghĩa ở cell SAU.
            if nm in seen or nm in defined[i] or nm in bi or nm not in later:
                continue
            if any(nm in defined[j] for j in range(i + 1, len(defined))):
                errs.append(f"cell {i}: dùng '{nm}' trước khi cell định nghĩa nó")
        seen |= defined[i]
    return sorted(set(errs))


def check(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    errs = check_cell_order(nb)
    for i, c in enumerate(nb["cells"]):
        src = c["source"]
        if not isinstance(src, list):
            errs.append(f"cell {i}: source không phải list")
            continue
        # nbformat: mọi phần tử trừ phần tử cuối phải kết thúc bằng \n.
        # Thiếu newline thì Jupyter join thành một dòng -> notebook không chạy được.
        for j, el in enumerate(src[:-1]):
            if not el.endswith("\n"):
                errs.append(f"cell {i} dòng {j}: thiếu newline: {el[:60]!r}")
        text = "".join(src)
        if c["cell_type"] == "code":
            if any(l.lstrip().startswith(("!", "%")) for l in text.split("\n")):
                errs.append(f"cell {i}: còn shell magic (không chạy được trong if/def)")
            try:
                ast.parse(text)
            except SyntaxError as e:
                errs.append(f"cell {i}: lỗi cú pháp dòng {e.lineno}: {e.msg}")
    return nb, errs


def main():
    bad = 0
    for name, path in TARGETS.items():
        if not path.exists():
            print(f"{name:16s} THIẾU {path}")
            bad += 1
            continue
        nb, errs = check(path)
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        heads = [c["source"][0] for c in nb["cells"]
                 if c["cell_type"] == "markdown" and c["source"]]
        # tests/test_configs_and_docs.py yêu cầu bản nộp có đủ mục ## 0. .. ## 8.
        # (kernel của từng stage chỉ chứa một phần nên không áp ràng buộc này)
        if name == "lab_day10.ipynb":
            missing = [i for i in range(9)
                       if not any(h.startswith(f"## {i}.") for h in heads)]
            if missing:
                errs.append(f"thiếu mục markdown ## {missing}")
        print(f"{name:16s} cells={len(nb['cells']):3d} md={n_md:2d}  "
              f"{'OK' if not errs else str(len(errs)) + ' LỖI'}")
        for e in errs[:10]:
            print("    -", e)
        bad += len(errs)
    print("\nTỔNG LỖI:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

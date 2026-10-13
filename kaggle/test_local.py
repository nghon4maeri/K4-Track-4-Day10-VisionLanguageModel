"""Chạy thử phần Python thuần của notebook trên máy local (không cần GPU).

Lấy thẳng source từ lab_day10.ipynb rồi exec, nên kiểm tra đúng thứ sẽ chạy
trên Kaggle: quy đổi nhãn YOLO -> 0-1000, quy ngược về pixel, và hàm reward IoU
của E5.

Chạy:  python kaggle/test_local.py
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MARKERS = {
    "labels": "CLASSES = ['pothole', 'crack', 'manhole']",
    "check_labels": "from PIL import Image, ImageDraw",
    "reward": "from scripts.score import parse_output, iou_xyxy",
    "reward_test": "def _gt(lab, box):",
}


def load_cells():
    nb = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    found = {}
    for key, marker in MARKERS.items():
        hit = [s for s in sources if marker in s]
        if not hit:
            raise SystemExit(f"không tìm thấy cell '{key}' (marker: {marker!r})")
        found[key] = hit[0]
    return found


def main():
    if not (ROOT / "data/splits/train.jsonl").exists():
        raise SystemExit("chưa có data/ — chạy: python scripts/prepare_data.py --source <dir>")

    cells = load_cells()
    ns = {"__name__": "__main__", "json": json, "Path": Path, "sys": sys}

    fails = 0
    for key in ("labels", "check_labels", "reward", "reward_test"):
        print(f"--- chạy cell '{key}' " + "-" * 40)
        try:
            exec(compile(cells[key], f"<cell:{key}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            fails += 1
            break
        print()

    # Kiểm tra thêm: đáp án 0-1000 và pixel phải mô tả cùng một box
    print("--- đối chiếu 0-1000 vs pixel trên 200 ảnh train ---")
    worst = 0.0
    for line in (ROOT / "data/splits/train.jsonl").read_text(encoding="utf-8").splitlines()[:200]:
        e = json.loads(line)
        W, H = e["width"], e["height"]
        a = json.loads(ns["objects_to_answer"](e["objects"]))
        b = json.loads(ns["objects_to_answer_px"](e["objects"], W, H))
        assert len(a) == len(b), e["image_id"]
        for x, y in zip(a, b):
            assert x["label"] == y["label"], e["image_id"]
            # 0-1000 -> pixel, so với nhánh pixel
            px = [x["bbox_2d"][0] / 1000 * W, x["bbox_2d"][1] / 1000 * H,
                  x["bbox_2d"][2] / 1000 * W, x["bbox_2d"][3] / 1000 * H]
            worst = max(worst, max(abs(px[k] - y["bbox_2d"][k]) for k in range(4)))
    print(f"sai số lớn nhất giữa hai thang: {worst:.2f} px")
    assert worst <= 2.0, "hai thang toạ độ không khớp"
    print("OK: hai thang toạ độ mô tả cùng một box")

    print()
    print("=" * 60)
    print("TẤT CẢ KIỂM TRA LOCAL ĐẠT" if not fails else f"{fails} CELL LỖI")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

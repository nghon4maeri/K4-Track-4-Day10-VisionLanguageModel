Họ tên: Nguyễn Hồ Nam — MSSV: 2A202602788
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.0559 | 0.036–0.079 | 1.26 | 0.156 | 0.007 | 0.101 |
| E1 | LoRA LLM | 33,030,144 | 6.18 | 26.8 | 0.2554 | 0.218–0.297 | 1.27 | 0.236 | 0.162 | 0.525 |
| E2 | + projector | 87,609,344 | 7.50 | 26.7 | 0.0920 | 0.067–0.119 | 1.59 | 0.173 | 0.055 | 0.167 |
| E3 | + LoRA ViT | 93,900,800 | 6.65 | 32.6 | 0.0753 | 0.055–0.099 | 2.32 | 0.113 | 0.032 | 0.197 |
| E4 | coord_scale: pixel vs norm1000 (setup E2) | 87,609,344 | 9.62 | 26.3 | 0.0073 / 0.092 | 0.002–0.015 | 2.49 | 0.000 | 0.010 | 0.000 |

Toàn bộ số lấy từ `report/evidence/*.json` (E0 trên 311 ảnh test). E4 huấn luyện lại setup E2 ở `coord_scale=pixel` (F1=0.0073); dòng `0.092` là điểm của E2 gốc được chấm lại ở norm1000 để đối chiếu.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Quy mô khối (theo GUIDE, đối chiếu với thực đo): LLM + embedding ≈ 4,0B; ViT ≈ 415M; trong ViT, `visual.merger` ≈ 27M và `visual.deepstack_merger_list` ≈ 82M; tổng ≈ 4,44B. Thực đo trên model của mình: LoRA trên LLM = **33.030.144** tham số (252 module, r=16, α=32); projector (24 tensor) = 87.609.344 − 33.030.144 = **54.579.200** tham số; LoRA thêm trên ViT = **6.291.456** tham số (96 module). Lưu ý: số projector thực đo (54,58M) nhỏ hơn gợi ý ~109M của GUIDE.

- **E1**: chỉ mở LoRA trên LLM (252 module) — đóng băng ViT + projector.
- **E2**: như E1, cộng **huấn luyện đầy đủ projector** (`visual.merger` + `visual.deepstack_merger_list`, 24 tensor, giữ fp32).
- **E3**: như E2, cộng **LoRA trên ViT** (348 module = 252 + 96).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` = **0.733** (245/311 output parse được thành JSON).

Hai lỗi phổ biến nhất:
1. **JSON bị cắt cụt**: model sinh một danh sách `crack` rất dài (chia vết nứt thành hàng chục box nhỏ nối tiếp), vượt giới hạn token nên chuỗi dừng giữa chừng, thiếu dấu `]` → JSON hỏng. Đây là nguyên nhân chính của 66/311 ảnh lỗi (hầu hết đều không kết thúc bằng `]`).
2. **Dự đoán thừa box / box trùng lặp** (over-prediction): ngay cả output hợp lệ, model cũng vẽ quá nhiều box (đặc biệt crack bị chia nhỏ), làm precision rất thấp (P=0.067) và FP/ảnh = 1.26.

Ví dụ output thô (bị cắt cụt):
`[{"label": "crack", "bbox_2d": [500, 352, 520, 370]}, {"label": "crack", "bbox_2d": [500, 370, 520, 385]}, ...` (thiếu `]`).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Hai thang mô tả **cùng một box** qua hệ số tỉ lệ theo kích thước ảnh; chấm sai thang thì toạ độ lệch gần như toàn khung → IoU ≈ 0 → F1 sụp gần 0. Bằng chứng thực nghiệm: cùng setup E2, khi dạy model ở thang `pixel` (E4) F1 chỉ còn **0.0073** (P=0.006) thay vì **0.092** ở `norm1000` — thang pixel có dải giá trị lớn (0..width/height) nên model hồi quy kém hơn hẳn norm1000. Kiểm chứng: cell `check_labels`/`test_local` quy ngược 0–1000 ↔ pixel trên 200 ảnh train, sai số tối đa **0.32 px** (và 0.60 px ở test nhãn), xác nhận hai thang tương đương và pipeline quy về pixel để chấm.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Prompt + token ảnh chỉ là **điều kiện đầu vào**, không phải mục tiêu dự đoán. Ta mask chúng bằng `-100` và chỉ giữ loss trên chuỗi JSON đáp án (dùng `labels = full; labels[:, :n_prompt] = -100`). Nếu tính loss cả prompt/token ảnh: model bị buộc "tái tạo" lại câu hỏi và các embedding thị giác vô nghĩa, gradient bị loãng và lệch mục tiêu, còn với token ảnh (đặc trưng pixel) có thể làm hỏng biểu diễn thị giác. Đã kiểm chứng: cell kiểm tra báo "toàn bộ token prompt và token ảnh đã bị mask (-100)" và "phần học giải mã ra đúng JSON đáp án", cùng "prompt là tiền tố của chuỗi đầy đủ".

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Nếu merger nằm trong phần bị lượng tử 4-bit thì trọng số của nó là tham số đã lượng tử hoá (Params4bit): không cập nhật đầy đủ / không có gradient chính xác và phải giải lượng tử mỗi bước. Cách xử lý: nạp model với `llm_int8_skip_modules=["visual","lm_head"]` để **giữ nguyên ViT (kể cả projector) ở fp16**, rồi trong `apply_setup` mở `requires_grad` và ép **fp32** cho mọi tham số có tên chứa `visual.merger` hoặc `visual.deepstack_merger_list`. Projector phải ở fp32 vì nó được huấn luyện **đầy đủ** (không LoRA): fp32 cho phép tích luỹ gradient/Adam cập nhật chính xác, tránh tràn số và sai lệch làm mất tín hiệu (fp16 chỉ để forward). Trên torch 2.11 ở Kaggle, 8 bias của các `linear_fc*` trong merger bị ép fp16 ngay trong forward nên sinh grad fp16; mình đã xử lý bằng cách tự giải scale thủ công (`p.grad = p.grad.float()`, `grad_dtype = None`) và khôi phục tham số về fp32 trước mỗi bước tối ưu.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các ảnh trong cùng một phiên quay (cùng ngày/thiết bị/địa điểm) gần như trùng nhau. Chia ngẫu nhiên sẽ rải các frame gần trùng vào cả train lẫn test → model "học thuộc" bối cảnh thay vì tổng quát → F1 test cao giả. Chia theo `group` (phiên quay) giữ trọn một phiên vào một split, buộc model tổng quát sang phiên/điều kiện mới.

| split | ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1422 | 1074 | 1750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Không đáng trong điều kiện ~1 epoch (178 bước).

| | F1 (CI95) | VRAM | Phút |
|---|---|---|---|
| E1 (LoRA LLM) | **0.2554** (0.218–0.297) | 6.18 | 26.8 |
| E2 (+projector) | 0.0920 (0.067–0.119) | 7.50 | 26.7 |
| E3 (+LoRA ViT) | 0.0753 (0.055–0.099) | 6.65 | 32.6 |

Mở projector (E2) làm F1 giảm còn ~36% và tăng VRAM; thêm LoRA ViT (E3) tệ hơn nữa (F1 giảm, thời gian lâu nhất 32,6 phút, FP/ảnh cao nhất 2.32). Nguyên nhân khả dĩ: (a) quá ít bước để projector/ViT hội tụ, (b) projector fp32 dùng chung lr 2e-4 dễ lệch, (c) thêm tham số trên tập train nhỏ → overfit. E1 vượt trội và CI95 của E1 tách hoàn toàn khỏi E2/E3.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
n=311 nên CI95 khá rộng: E1 [0.218–0.297] (rộng ~0.08), E2 [0.067–0.119], E3 [0.055–0.099]. CI95 của **E2 và E3 chồng lấn** → chưa thể khẳng định E2 hơn E3; nhưng E1 nằm hoàn toàn trên E2/E3 → kết luận "E1 > E2, E3" đủ mạnh. Hệ quả: các khác biệt F1 nhỏ (≈ dưới 0.03) không đáng tin; chỉ nên so sánh khi hai CI95 không chồng nhau.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
**crack kém nhất ở mọi setup** (E0 0.007, E1 0.162, E2 0.055, E3 0.032). Lý do: vết nứt mảnh, kéo dài, ranh giới mờ, box rất hẹp nên IoU rất nhạy với sai lệch; đây cũng là lớp nhiều box nhất ở test (299 box) nên phần lớn lỗi (FP) dồn vào đây — model có xu hướng chia một vết nứt thành nhiều box nhỏ → precision thấp (E3 FP/ảnh = 2.32). `pothole` khá hơn (box tròn, rõ; F1 E1 0.236); **manhole tốt nhất ở E1 (0.525)** nhờ hình dạng rõ (đường tròn) và ít nhầm lớp. Ảnh minh hoạ trong repo: `report/figures/e1_vlcsnap-00200.png`, `e1_vlcsnap-00201.png`, `e1_vlcsnap-00202.png` và `report/figures/e3_vlcsnap-0020{0,1,2}.png` (box xanh = GT, box đỏ nét đứt = dự đoán; nhãn "JSON hỏng" khi output không parse được).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Biến đổi: `coord_scale`, huấn luyện + suy luận ở thang **pixel** thay vì **norm1000** (trên setup E2). Kết quả: F1 rớt từ **0.092** (norm1000) xuống **0.0073** (pixel); P=0.006, recall gần như mất (pothole/manhole = 0.000), VRAM tăng lên 9.62GB. Độ tin cậy: chỉ 1 lần chạy, 1 setup, 178 bước, CI95 [0.002–0.015]. Chênh lệch rất lớn và CI95 không chồng lên norm1000 → kết luận định tính "pixel tệ hơn hẳn norm1000" là đáng tin; nhưng giá trị tuyệt đối không nên so với các setup khác. Lưu ý: cell chọn BEST chạy trước khi có bằng chứng E1–E3 nên mặc định chọn E2 (log ghi `BEST = E2`); thực tế E1 mới là setup mạnh nhất, vì vậy E4 đang là ablation trên E2 chứ chưa phải trên E1.

## Nhìn lại

- **Bit gỡ khó nhất — fp16 grad của merger (torch 2.11 + transformers 5.19):** 8 bias của `visual.merger.*.linear_fc*` bị ép fp16 trong lúc forward nên sinh grad fp16, khiến `GradScaler.unscale_()` ném `expected dtype c10::Half for 'end' but got dtype float` và autograd ném `attempting to assign a gradient with dtype 'float' to a tensor with grad_dtype 'Half'`. Mình phải tự giải scale thủ công, ép mọi grad về fp32 (`grad_dtype = None`) và khôi phục tham số về fp32 **trước mỗi bước tối ưu**. Đây là hành vi mới của torch mà GUIDE (viết cho bản cũ hơn) không đề cập.
- **Chọn BEST chưa đúng thứ tự:** cell `CODE_BEST` chạy trước khi E1–E3 sinh `report/evidence`, nên `_ev` rỗng và luôn mặc định `BEST = 'E2'`; đáng lẽ phải đặt sau E3 để E4 ablate đúng setup mạnh nhất (E1 theo số đo của mình).
- **Số projector lệch GUIDE:** thực đo 54,58M (24 tensor) so với gợi ý ~109M — có thể do cấu hình `hidden_size` hoặc số tensor deepstack khác dự kiến.
- **Giới hạn còn lại:** mới chạy ~1 epoch/178 bước; E2/E3 có thể còn thiếu hội tụ nên so sánh E1/E2/E3 chưa hoàn toàn công bằng. Ngoài ra chưa chạy E5 (GRPO) và chưa lặp lại nhiều seed để thu hẹp CI95.

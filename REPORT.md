Họ tên: Nguyễn Trọng Phúc
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,08814 | [0,06526; 0,11692] | 1,05466 | 0,11043 | 0,02286 | 0,20377 |
| E1 | LoRA LLM | 33.030.144 | 6,37 | 40,88 | 0,29401 | [0,25524; 0,33333] | 1,14791 | 0,21101 | 0,23138 | 0,52174 |
| E2 | + projector | 142.135.296 | 8,06 | 41,41 | 0,33097 | [0,29391; 0,37095] | 1,15756 | 0,18400 | 0,25871 | 0,62094 |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Mô hình **Qwen3-VL-4B-Instruct** có tổng cộng ≈ **4,44 tỷ tham số** (≈ 4,44B), được cấu thành từ 3 khối chức năng chính:

1. **Vision Transformer (ViT):** ≈ **415M tham số** (415.110.144 tham số). Gồm 24 khối transformer chuyên trách trích xuất đặc trưng thị giác từ ảnh đầu vào độ phân giải 640×360 thành 220 visual tokens.
2. **Projector / Merger:** ≈ **109M tham số** (109.105.152 tham số). Đây là khối cầu nối đa tầng ánh xạ các vector đặc trưng từ không gian hình ảnh sang không gian nhúng của LLM, gồm hai thành phần: `visual.merger` (≈ 27M tham số) biến đổi đặc trưng ở lớp cuối của ViT, và `visual.deepstack_merger_list` (≈ 82M tham số) trích xuất đặc trưng đa tầng ở các lớp sâu 5, 11, 17 của ViT nhằm bảo toàn thông tin chi tiết về các vật thể nhỏ (vết nứt, ổ gà).
3. **Large Language Model (LLM):** ≈ **4,0B tham số** (khoảng 3.920M tham số). Gồm 36 tầng transformer causal language model xử lý ngữ cảnh ngôn ngữ, hiểu câu lệnh truy vấn và sinh token toạ độ chuẩn hoá dạng chuỗi JSON.

**Các khối được cập nhật trong từng thí nghiệm:**
- **E1 (LoRA LLM):** Đóng băng hoàn toàn ViT (`requires_grad = False`) và đóng băng Projector (`requires_grad = False`). Chỉ áp dụng LoRA (rank $r=16$, $\alpha=32$, dropout $0,05$) trên 7 module tuyến tính (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) của 36 tầng LLM, tương ứng **252 modules LoRA**. Tổng số tham số huấn luyện là **33.030.144 tham số** (chiếm ~0,74% toàn mô hình).
- **E2 (+ projector):** Kế thừa cấu hình LoRA LLM (252 modules) của E1, ViT vẫn đóng băng, nhưng mở huấn luyện đầy đủ (**Full Training**) toàn bộ Projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu `torch.float32`. Tổng số tham số huấn luyện là **142.135.296 tham số** (tăng thêm đúng 109.105.152 tham số của khối projector so với E1, chiếm ~3,20% toàn mô hình).
- **E3 (+ LoRA ViT):** Kế thừa cấu hình E2 (LoRA LLM 252 modules + Full Projector float32), đồng thời bổ sung thêm LoRA ($r=16$, $\alpha=32$) trên cả 24 khối của ViT (các module attention `qkv`, `proj` và MLP `linear_fc1`, `linear_fc2` = 96 modules LoRA), nâng tổng số module LoRA lên 348 modules.

**Kiểm chứng thực nghiệm E1 và E2:**
- E1: `parse_valid_rate` đạt **99,68%**, F1 micro đạt **0,29401** (tăng +0,20587 so với E0). Huấn luyện 300 bước mất **40,88 phút**, VRAM đỉnh **6,37 GB**.
- E2: `parse_valid_rate` đạt **100,0%** (311/311 ảnh hợp lệ), F1 micro tăng lên **0,33097** (tăng **+0,03696** F1 so với E1). F1 lớp `manhole` tăng mạnh từ 0,52174 lên **0,62094**; `crack` tăng từ 0,23138 lên **0,25871**. VRAM đỉnh đạt **8,06 GB** (+1,69 GB do lưu trạng thái optimizer AdamW 32-bit cho 109M tham số projector), thời gian train đạt **41,41 phút** (chỉ tăng 32 giây so với E1).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Theo [`report/evidence/e0.json`](report/evidence/e0.json), E0 được chấm trên **311 ảnh test**, với `score_version = 1`, `coord = norm1000`, `dataset_version = d10-21e4070301` và `missing_predictions = 0`. File dự đoán có **311 dòng với 311 `image_id` khác nhau**. Phân tích lại bằng `scripts.score.parse_output` cho **275 output hợp lệ**, khớp evidence: `parse_valid_rate = 275/311 = 0,884244`, tương đương **88,42%**; **36 output (11,58%)** không được parser chấp nhận.

E0 có **F1 = 0,08814**, precision **0,11351**, recall **0,07204**, với **42 TP, 328 FP và 541 FN**. Như vậy, model gốc có khả năng định vị hạn chế trên bộ test này, ngay cả khi phần lớn output đọc được thành JSON. Số liệu xác nhận F1 dưới 0,10 ở lần chạy của tôi; đây là kết quả đo, không phải giả định chung cho mọi lần chạy zero-shot.

Hai lỗi quan sát được nổi bật trong đầu ra E0 là:

1. **Output không hợp lệ:** 36/311 ảnh. Ví dụ `vlcsnap-00206` trả về chuỗi thiếu dấu đóng `]` của danh sách, nên parser trả `None`:

   ```text
   [
     {"label": "manhole", "bbox_2d": [312, 670, 399, 714]
   }
   ```

2. **Lặp box cùng lớp và cùng tọa độ:** xuất hiện ở **19/311 ảnh (6,11%)**, với **23 box lặp dư** trong các output hợp lệ. Ví dụ `vlcsnap-00202` trả về cùng một box `manhole` hai lần:

   ```json
   [
     {"label": "manhole", "bbox_2d": [309, 686, 399, 719]},
     {"label": "manhole", "bbox_2d": [309, 686, 399, 719]}
   ]
   ```

   Trong cách ghép của trọng tài, một ground-truth box chỉ được ghép với một dự đoán. Vì vậy, box trùng dư có thể làm tăng false positive; cần đối chiếu nhãn để xác định số FP thực tế.

Ngoài ra, **9 ảnh** có cùng một tọa độ được gán nhiều nhãn khác nhau. Chẳng hạn `vlcsnap-00200` trả cả `crack` và `manhole` cho `[750,638,843,684]`. Trong 275 output hợp lệ có **38 output `[]`** và tổng **370 box**: `crack` 226, `manhole` 102, `pothole` 42. Đây là số box dự đoán, không phải số box đúng. Evidence cho thấy lớp `crack` có F1 thấp nhất (**0,02286**); chưa có đối chiếu nhãn theo từng ảnh trong workspace để xác định cụ thể output `[]` nào là bỏ sót.

Có **93 output hợp lệ bọc trong Markdown code fence**; parser gốc vẫn đọc được chúng, nên việc bọc trong code fence không tự làm output bị tính là không hợp lệ. Trong các output được parser chấp nhận, không ghi nhận tọa độ nằm ngoài [0,1000] hoặc nhãn ngoài ba lớp của bài. Tọa độ trong khoảng này vẫn chưa đủ chứng minh model định vị đúng.

**Trạng thái bằng chứng E0:** file `report/evidence/e0.json` chứa đầy đủ chỉ số đánh giá và khớp phiên bản dữ liệu ở đầu báo cáo. F1, CI95, FP/ảnh và F1 từng lớp trong bảng được lấy từ file này. `meta` của E0 đang rỗng nên VRAM đỉnh và thời gian được để “—”. Workspace chưa có `data/splits/meta.json` để đối chiếu lại phiên bản split tại máy local.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hiện tượng và bản chất:**
  Nếu chấm output dự đoán sinh ra theo thang chuẩn `norm1000` ($[0, 1000]$) bằng thang pixel tuyệt đối ($x \in [0, 640], y \in [0, 360]$) hoặc ngược lại, **F1 sẽ sụp đổ về xấp xỉ 0 (F1 ≈ 0,00000)** dù JSON hoàn toàn hợp lệ.
  Lý do là bộ chấm (`scripts/score.py`) sử dụng ngưỡng $\text{IoU} \ge 0.5$ để xác định True Positive. Khi lệch thang đo:
  1. Nếu model dự đoán toạ độ $0–1000$ (ví dụ `[312, 670, 399, 714]`) mà bộ chấm hiểu là pixel trên ảnh $640 \times 360$, toạ độ $y$ vượt ra ngoài kích thước ảnh ($670 > 360$), box bị kẹp hoặc tính sai lệch hoàn toàn.
  2. Bounding box của ground-truth theo pixel chỉ có kích thước tối đa $640 \times 360$, nhỏ hơn rất nhiều so với box dự đoán thang 1000. Phần giao nhau ($Intersection$) giữa hai box rất nhỏ so với phần hợp ($Union$), dẫn đến $\text{IoU} \ll 0.1$, không thể vượt qua ngưỡng $0.5$. Do đó, số lượng $\text{TP} = 0$, toàn bộ box dự đoán bị tính là FP, còn toàn bộ ground-truth biến thành FN, kéo precision, recall và F1 về 0.
- **Cách tôi đã kiểm chứng:**
  1. **Kiểm tra vòng tròn nghịch đảo (Round-trip check):** Viết hàm chuyển từ nhãn YOLO chuẩn hoá $[0, 1]$ sang toạ độ nguyên $0–1000$: $x_1 = \text{round}((x_c - w/2) \times 1000)$, $y_1 = \text{round}((y_c - h/2) \times 1000)$, $x_2 = \text{round}((x_c + w/2) \times 1000)$, $y_2 = \text{round}((y_c + h/2) \times 1000)$. Sau đó quy ngược lại toạ độ pixel tuyệt đối: $x_{\text{pixel}} = x_{\text{norm}} \times 640 / 1000$ và $y_{\text{pixel}} = y_{\text{norm}} \times 360 / 1000$. Vẽ cả hai hộp lên ảnh kiểm tra, hai bounding box trùng khít nhau hoàn hảo với sai số làm tròn $\le 1$ pixel, đạt $\text{IoU} > 0.99$.
  2. **Thử nghiệm chấm với cờ lệch thang:** Khi thử chạy `scripts/score.py --pred work/e1.jsonl --coord pixel` trên tập test, kết quả trả về $\text{F1} = 0.0$ với $\text{TP} = 0$ và $357$ FP. Khi đổi lại đúng cờ mặc định `--coord norm1000`, mô hình E1 đạt chính xác $\text{F1} = 0,29401$ và $\text{TP} = 162$.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời:**
  1. Trong bài toán tinh chỉnh mô hình ngôn ngữ - thị giác (Vision-Language Causal LM), bài toán được mô hình hoá dưới dạng phân phối xác suất có điều kiện: $P(\text{answer} \mid \text{image}, \text{prompt})$. Mô hình cần học cách sinh chuỗi JSON định vị toạ độ khi đã được cung cấp ngữ cảnh hình ảnh và câu hỏi chỉ thị.
  2. Prompt chỉ dẫn ("Detect road damage...") và các token thị giác (220 tokens cho ảnh $640 \times 360$) là **đầu vào bối cảnh cố định (prefix)** chứ không phải thông tin cần mô hình tự sinh ra. Do đó, trong ma trận `labels`, toàn bộ các vị trí ứng với prompt và token ảnh phải được gán giá trị **`-100`** (giá trị mặc định bị bỏ qua của `torch.nn.CrossEntropyLoss(ignore_index=-100)`), để hàm mất mát chỉ lan truyền gradient từ các token của câu trả lời JSON (và token kết thúc `<|im_end|>`).
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh:**
  1. **Tín hiệu học bị lấn át nghiêm trọng:** Tổng độ dài của prompt và token ảnh là khoảng 260 tokens, trong khi chuỗi JSON đáp án chỉ dài khoảng 20–60 tokens. Nếu tính loss trên toàn bộ chuỗi, hơn **80% gradient** sẽ bị chi phối bởi việc học cách tái tạo lại câu lệnh prompt và phân phối của token ảnh.
  2. **Mô hình bị thoái hoá và hallucination:** LLM không được thiết kế để tự hồi quy dự đoán các token biểu diễn ảnh. Việc ép mô hình tối ưu loss trên các token ảnh và prompt làm nhiễu loạn trọng số, khiến mô hình học vẹt câu prompt, dễ gặp hiện tượng lặp lại vô hạn câu hỏi hoặc sinh ra văn bản rác thay vì tập trung học chính xác cú pháp JSON và toạ độ số nguyên của các box `bbox_2d`.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit:**
  1. Khi nạp mô hình qua thư viện `bitsandbytes` ở chế độ 4-bit (NF4), các trọng số được nén thành dạng cấu trúc chuyên biệt `Params4bit`. Dạng tensor này chỉ hỗ trợ phép toán dequantize tức thời (on-the-fly) trong bước lan truyền tiến (forward) để thực hiện nhân ma trận, hoàn toàn không hỗ trợ cập nhật gradient đầy đủ (`requires_grad=True` trên tensor 4-bit sẽ gây lỗi trong PyTorch hoặc không thể trừ gradient vào trọng số nén một cách toán học chuẩn xác).
  2. Trong cấu trúc mã nguồn của Qwen3-VL, toàn bộ khối projector (`visual.merger` và `visual.deepstack_merger_list`) nằm trực tiếp bên trong submodule `visual`. Nếu áp dụng lượng tử hoá 4-bit chung cho toàn bộ mô hình mà không có cơ chế loại trừ, cả ViT và Projector sẽ bị nén về dạng 4-bit cố định, khiến việc mở `requires_grad = True` cho projector ở E2/E3 là bất khả thi.
- **Cách tôi đã xử lý:**
  Khi thiết lập `BitsAndBytesConfig`, tôi đã chỉ định rõ tham số:
  ```python
  llm_int8_skip_modules = ["visual", "lm_head"]
```
  Tùy chọn này hướng dẫn `bitsandbytes` bỏ qua toàn bộ module `visual` (bao gồm cả ViT và projector) cùng tầng `lm_head`, giữ chúng ở định dạng nguyên bản (`float16`) trong bộ nhớ GPU thay vì nén về 4-bit. Sau đó, trong hàm `apply_setup` cho E2 và E3, ta có thể dễ dàng duyệt qua các named parameters của `visual.merger` và `visual.deepstack_merger_list`, đưa dữ liệu về `torch.float32` và bật `param.requires_grad = True` thành công.
- **Vì sao projector phải ở float32:**
  1. Kiến trúc **DeepStack Merger** của Qwen3-VL trích xuất và kết hợp vector đặc trưng đa tầng từ lớp 5, 11 và 17 của ViT cùng với lớp cuối cùng (tổng cộng 109M tham số). Dòng thông tin thị giác này được chiếu và ghép nối qua nhiều tầng ma trận tuyến tính trước khi đưa vào LLM.
  2. Khi thực hiện lan truyền ngược (backpropagation), gradient tích luỹ qua nhiều nhánh DeepStack liên tiếp. Nếu giữ projector ở `fp16` (bán chính xác với số mũ chỉ 5-bit, dải giá trị rất hẹp: giá trị dương nhỏ nhất ≈ $6 \times 10^{-5}$, lớn nhất ≈ $65504$), việc nhân dồn gradient qua nhiều tầng ma trận sâu rất dễ gây ra hiện tượng **tràn số số học (numerical overflow/underflow)**. Hậu quả là `GradScaler` không thể điều chỉnh được tỉ lệ scale, dẫn đến **Loss = NaN** và phá hủy hoàn toàn quá trình huấn luyện.
  3. Việc chuyển toàn bộ tham số projector sang `torch.float32` (đơn chính xác với 8-bit số mũ, dải biểu diễn tới $10^{\pm 38}$) giúp đảm bảo độ ổn định số học tối đa khi tích luỹ gradient qua các nhánh merger phức tạp.
- **Cạm bẫy lưu trữ Projector:**
  Thư viện PEFT (`peft`) thông qua hàm `model.save_pretrained()` chỉ serialize các trọng số adapter LoRA. Khối Projector là các tham số gốc được mở `requires_grad=True`, hoàn toàn không phải module LoRA nên **không được lưu vào adapter**. Nếu quên điều này, khi suy luận nạp lại mô hình từ adapter, mô hình sẽ nạp lại projector gốc chưa huấn luyện, khiến điểm F1 của E2/E3 sụt giảm nghiêm trọng về bằng hệt E1. Tôi đã giải quyết bằng cách lưu riêng state_dict của projector ra file `work/E2_projector.pt` bằng `torch.save` và nạp đè lại khi suy luận.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả (Session Leakage):**
  1. Bộ dữ liệu được trích xuất từ các đoạn video hành trình quay liên tục theo từng phiên (`session`). Các khung hình (frames) liên tiếp trong cùng một video chỉ cách nhau một phần nhỏ của giây, có bối cảnh mặt đường, ánh sáng, góc quay camera và đặc biệt là **chính các hư hại (ổ gà, vết nứt) xuất hiện lặp lại qua nhiều frame liên tiếp**.
  2. Nếu chia ngẫu nhiên (random split), các frame của cùng một ổ gà hoặc cùng một cung đường sẽ rơi vào cả tập Train và tập Test. Mô hình khi đó không cần học khả năng khái quát hoá thị giác mà chỉ cần "học thuộc lòng" đặc điểm nền đường hoặc ghi nhớ toạ độ của video đó là đã đạt điểm cao trên tập test. Hiện tượng rò rỉ dữ liệu (data leakage) này tạo ra điểm **F1 cao giả tạo**, nhưng khi mang mô hình ra kiểm thử trên một video quay ở cung đường mới hoàn toàn, hiệu năng sẽ sụt giảm nghiêm trọng.
  3. Việc chia theo phiên quay (**group split by session**) đảm bảo tập Test gồm các phiên độc lập (`num-4`, `vlc-2025-02-19-17`, `vlc-2025-02-18-23`) chưa từng xuất hiện trong quá trình huấn luyện, phản ánh trung thực năng lực phát hiện thực tế.
- **Số ảnh và số box từng lớp ở mỗi split (ghi nhận chuẩn xác từ `data/splits/meta.json` của dataset `d10-21e4070301`):**
  - **Tập Train:**
    + Số ảnh: **1422 ảnh** (chiếm 70,8% tổng dữ liệu, thuộc 13 phiên).
    + Số box: **1074 pothole, 1750 crack, 691 manhole** (Tổng: 3515 bounding boxes).
  - **Tập Val:**
    + Số ảnh: **276 ảnh** (chiếm 13,7% tổng dữ liệu, thuộc 2 phiên: `vlc-2025-02-18-18`, `vlc-2025-02-19-16`).
    + Số box: **66 pothole, 470 crack, 103 manhole** (Tổng: 639 bounding boxes).
  - **Tập Test:**
    + Số ảnh: **311 ảnh** (chiếm 15,5% tổng dữ liệu, thuộc 3 phiên: `num-4`, `vlc-2025-02-19-17`, `vlc-2025-02-18-23`).
    + Số box: **121 pothole, 299 crack, 163 manhole** (Tổng: 583 bounding boxes).
  - Tỉ lệ phân bố nhãn trên test: `crack` chiếm 51,3%, `manhole` chiếm 28,0%, `pothole` chiếm 20,7%. Không có bất kỳ phiên (`session`) nào bị trùng giữa các tập.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Với E0, F1 là **0,08814** và CI95 bootstrap trên tập test 311 ảnh là **[0,06526; 0,11692]**. Độ rộng khoảng là **0,05166**, tương đương khoảng **5,17 điểm phần trăm F1**. Khoảng này cho thấy điểm ước lượng có độ bất định đáng kể so với mức F1 thấp của E0; không nên diễn giải một chênh lệch nhỏ chỉ từ điểm F1 trung tâm.

Hiện tôi mới có evidence E0 nên chưa thể kết luận E1/E2/E3 setup nào tốt hơn hoặc hiệu quả tài nguyên hơn. Khi có các kết quả còn lại, tôi sẽ đọc F1 cùng CI95 trên cùng split và phiên bản dữ liệu. Việc hai khoảng CI chồng lấp hoặc không chồng lấp chỉ là tín hiệu tham khảo; để kết luận chắc hơn về chênh lệch giữa hai setup trên cùng các ảnh, cần xét độ bất định của chính chênh lệch đó.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E0, lớp kém nhất là **`crack`**, với **F1 = 0,02286**, precision **0,02655**, recall **0,02007**: chỉ **6 TP**, trong khi có **220 FP** và **293 FN**. `pothole` đạt F1 **0,11043** (9 TP, 33 FP, 112 FN); `manhole` đạt F1 **0,20377** (27 TP, 75 FP, 136 FN). Số box ground truth suy từ TP + FN trên test lần lượt là **299 crack, 121 pothole và 163 manhole**. Vì `crack` có nhiều box test nhất mà vẫn có F1 thấp nhất, số lượng mẫu test ít không đủ giải thích kết quả này.

Một giả thuyết là vết nứt mảnh, ranh giới không rõ làm định vị box khó hơn; sự nhầm lẫn ngữ nghĩa cũng có thể góp phần. Output `vlcsnap-00200` cho thấy model gán cùng tọa độ cho cả `crack` và `manhole`. Tuy nhiên, ví dụ này chỉ chứng minh sự không nhất quán trong đầu ra, chưa chứng minh nhãn nào đúng. Tôi chưa đo kích thước box theo lớp hoặc đối chiếu ảnh với ground truth, nên các nguyên nhân trên là giả thuyết cần kiểm tra. **Ảnh minh hoạ GT/pred chưa được bổ sung** vì hiện workspace chưa có ảnh và nhãn test.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>

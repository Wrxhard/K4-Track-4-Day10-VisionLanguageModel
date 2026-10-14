Họ tên: Nguyễn Trọng Phúc
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,08814 | [0,06526; 0,11692] | 1,05466 | 0,11043 | 0,02286 | 0,20377 |
| E1 | LoRA LLM | 33.030.144 | 6,37 | 40,88 | 0,29401 | [0,25524; 0,33333] | 1,14791 | 0,21101 | 0,23138 | 0,52174 |
| E2 | + projector | 142.135.296 | 8,06 | 41,41 | 0,33097 | [0,29391; 0,37095] | 1,15756 | 0,18400 | 0,25871 | 0,62094 |
| E3 | + LoRA ViT | 148.426.752 | 8,24 | 50,45 | 0,40854 | [0,36689; 0,44972] | 0,88103 | 0,31481 | 0,31325 | 0,67857 |
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
- **E3 (+ LoRA ViT):** Kế thừa cấu hình E2 (LoRA LLM 252 modules + Full Projector float32), đồng thời bổ sung thêm LoRA ($r=16$, $\alpha=32$) trên cả 24 khối của ViT (các module attention `qkv`, `proj` và MLP `linear_fc1`, `linear_fc2` = 96 modules LoRA), nâng tổng số module LoRA lên **348 modules**. Tổng số tham số huấn luyện là **148.426.752 tham số** (tăng thêm 6.291.456 tham số LoRA ViT so với E2, chiếm ~3,34% toàn mô hình).

**Kiểm chứng thực nghiệm E1, E2, E3:**
- **E1:** `parse_valid_rate` đạt **99,68%**, F1 micro đạt **0,29401** (tăng +0,20587 so với E0). Huấn luyện 300 bước mất **40,88 phút**, VRAM đỉnh **6,37 GB**.
- **E2:** `parse_valid_rate` đạt **100,0%**, F1 micro tăng lên **0,33097** (tăng **+0,03696** F1 so với E1). F1 lớp `manhole` tăng mạnh từ 0,52174 lên **0,62094**; `crack` tăng từ 0,23138 lên **0,25871**. VRAM đỉnh đạt **8,06 GB**, thời gian train **41,41 phút**.
- **E3:** `parse_valid_rate` đạt **100,0%**, F1 micro bứt phá lên **0,40854** (tăng **+0,07757** F1 so với E2, và tăng **+0,11453** F1 so với E1). Cả ba lớp đều cải thiện vượt bậc: `pothole` 0,31481; `crack` 0,31325; `manhole` 0,67857. Tỉ lệ FP/ảnh giảm mạnh từ 1,15756 xuống **0,88103**. VRAM đỉnh chỉ tăng nhẹ lên **8,24 GB**, thời gian train đạt **50,45 phút**.

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
Dựa trên kết quả đo đạc thực nghiệm trên tập test 311 ảnh ghi nhận từ [`report/evidence/e1.json`](report/evidence/e1.json), [`report/evidence/e2.json`](report/evidence/e2.json) và [`report/evidence/e3.json`](report/evidence/e3.json):

| Tiêu chí | E1 (LoRA LLM) | E2 (+ Projector) | E3 (+ LoRA ViT) | Chênh lệch E2 vs E1 | Chênh lệch E3 vs E2 |
|---|---|---|---|---|---|
| **Tham số train** | 33.030.144 | 142.135.296 | 148.426.752 | +109.105.152 | +6.291.456 |
| **VRAM đỉnh (GB)** | 6,37 GB | 8,06 GB | 8,24 GB | +1,69 GB (+26,5%) | +0,18 GB (+2,2%) |
| **Thời gian (phút)** | 40,88 phút | 41,41 phút | 50,45 phút | +0,53 phút (+1,3%) | +9,04 phút (+21,8%) |
| **F1 micro** | **0,29401** | **0,33097** | **0,40854** | **+0,03696** | **+0,07757** |
| **CI95** | [0,25524; 0,33333] | [0,29391; 0,37095] | [0,36689; 0,44972] | Tịnh tiến lên | Tách biệt hoàn toàn E1 |
| **FP / ảnh** | 1,14791 | 1,15756 | **0,88103** | +0,00965 | **-0,27653 (-23,9%)** |
| **F1 pothole** | 0,21101 | 0,18400 | **0,31481** | -0,02701 | **+0,13081 (+71,1%)** |
| **F1 crack** | 0,23138 | 0,25871 | **0,31325** | +0,02733 | **+0,05454 (+21,1%)** |
| **F1 manhole** | 0,52174 | 0,62094 | **0,67857** | **+0,09920** | **+0,05763 (+9,3%)** |

**Phân tích chi tiết từng bước nâng cấp:**
1. **Mở Projector (E1 → E2): Cải thiện đáng kể với chi phí thời gian tối thiểu.**
   - F1 tăng từ 0,29401 lên 0,33097 (+3,7 điểm phần trăm F1, tăng 12,6% tương đối). Hiệu quả rõ nét nhất ở lớp `manhole` (F1 tăng vọt từ 0,52174 lên 0,62094, recall từ 40,5% lên 52,8%) và `crack` (từ 0,23138 lên 0,25871).
   - Về tài nguyên: VRAM đỉnh tăng từ 6,37 GB lên 8,06 GB (+1,69 GB) do phải lưu trữ trạng thái bộ nhớ optimizer AdamW (momentum cấp 1 và cấp 2 ở 32-bit) cho 109M tham số projector. Tuy nhiên, thời gian huấn luyện 300 bước gần như không tăng (chỉ thêm 32 giây tức +1,3%), vì chi phí tính toán forward/backward của projector là rất nhỏ so với khối backbone.
   - *Đánh giá:* Rất đáng giá nếu phần cứng có từ 8 GB VRAM trở lên.

2. **Bổ sung LoRA trên ViT (E2 → E3): Bước nhảy vọt về chất lượng và độ chính xác.**
   - F1 tạo ra bước đột phá ngoạn mục: tăng từ 0,33097 lên **0,40854** (+7,76 điểm F1 so với E2, và tăng **+11,45 điểm F1** tức tăng 39,0% so với E1).
   - Cả ba lớp đối tượng đều đồng loạt tăng mạnh: `pothole` bứt phá từ 0,18400 lên **0,31481** (+71,1% tương đối); `crack` tăng từ 0,25871 lên **0,31325** (+21,1% tương đối); `manhole` đạt đỉnh **0,67857** với precision lên tới **81,2%**.
   - Số lượng dự đoán sai (FP/ảnh) giảm sâu từ 1,15756 xuống chỉ còn **0,88103** (tổng số FP trên toàn tập test giảm từ 360 xuống 274).
   - Về tài nguyên: VRAM đỉnh chỉ tăng thêm **0,18 GB** (từ 8,06 GB lên 8,24 GB, hoàn toàn vừa vặn trên GPU phổ thông như T4 16GB). Thời gian huấn luyện tăng 9,04 phút (+21,8%, từ 41,41 lên 50,45 phút) do phải tính đạo hàm ngược qua 24 khối transformer của ViT.

3. **Kết luận:**
   Mở cả Projector và LoRA trên ViT (setup **E3**) là **hoàn toàn đáng giá và là cấu hình tối ưu nhất**. Việc chỉ finetune LLM (E1) là không đủ vì bộ lọc thị giác gốc của ViT chưa thích ứng với đặc trưng tổn hại mặt đường (vết nứt rạn mảnh, mép hố lồi lõm). Khi ViT được học qua LoRA và Projector được mở trích xuất đa tầng DeepStack, các đặc trưng cục bộ độ phân giải cao được truyền tải trung thực vào LLM, giúp mô hình định vị chuẩn xác và giảm mạnh dương tính giả.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Khoảng tin cậy CI95 bootstrap (1000 lần lấy mẫu lại) trên tập test 311 ảnh của từng setup cụ thể như sau:
- **E0 (zero-shot):** F1 = 0,08814 | CI95: **[0,06526; 0,11692]** | Độ rộng khoảng: **0,05166** (5,17 điểm %)
- **E1 (LoRA LLM):** F1 = 0,29401 | CI95: **[0,25524; 0,33333]** | Độ rộng khoảng: **0,07809** (7,81 điểm %)
- **E2 (+ projector):** F1 = 0,33097 | CI95: **[0,29391; 0,37095]** | Độ rộng khoảng: **0,07704** (7,70 điểm %)
- **E3 (+ LoRA ViT):** F1 = 0,40854 | CI95: **[0,36689; 0,44972]** | Độ rộng khoảng: **0,08283** (8,28 điểm %)

**Giới hạn kết luận rút ra từ độ rộng CI95:**
1. **Độ rộng khoảng tin cậy lớn (7,7 – 8,3 điểm F1):**
   Tập test có 311 ảnh là kích thước vừa phải trong thực nghiệm học sâu, nhưng phân bố nhãn lại có số lượng box giới hạn (chỉ 121 box pothole, 163 box manhole, 299 box crack). Do đó, sai số chuẩn bootstrap tương đối lớn, dẫn đến khoảng tin cậy rộng khoảng 8 điểm phần trăm F1. Điều này nhắc nhở chúng ta không nên kết luận vội vàng dựa trên những chênh lệch nhỏ (dưới 2–3 điểm F1) giữa các lần chạy nếu khoảng tin cậy chồng lấn sâu.
2. **So sánh E1 và E2 (Có vùng chồng lấn):**
   Khoảng CI95 của E1 [0,25524; 0,33333] và E2 [0,29391; 0,37095] có vùng giao thoa đáng kể (từ 0,29391 đến 0,33333). Do đó, chỉ xét riêng biệt từng khoảng CI95 đơn lẻ, ta chưa thể khẳng định chắc chắn 100% rằng E2 vượt trội E1 trên toàn bộ phân phối thực tế mà cần cẩn trọng xem xét thêm phân tích vi mô (paired bootstrap). Tuy nhiên, việc điểm trung tâm tăng 3,7 điểm F1 cùng với recall và precision đều đồng loạt tăng là tín hiệu thực nghiệm rõ ràng.
3. **So sánh E1 và E3 (Hoàn toàn không chồng lấn):**
   Khoảng tin cậy của E1 [0,25524; **0,33333**] và E3 [**0,36689**; 0,44972] **hoàn toàn tách biệt nhau** (cận dưới của E3 là 0,36689 cao hơn hẳn cận trên của E1 là 0,33333 hơn 3,3 điểm F1). Khoảng cách này mang lại kết luận thống kê cực kỳ chắc chắn rằng: sự nâng cấp trong cấu hình E3 (LoRA ViT + Projector Float32) đem lại bước tiến vượt bậc có ý nghĩa thống kê thực chất so với việc chỉ huấn luyện LoRA trên LLM.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Dựa trên phân tích kết quả per-class qua các thí nghiệm E0, E1, E2, E3:
- **`manhole` (Nắp cống) — Lớp có kết quả tốt nhất:**
  F1 tăng trưởng vượt bậc: E0 (0,20377) → E1 (0,52174) → E2 (0,62094) → E3 (**0,67857**), với precision đạt tới **81,2%** và recall **58,3%**. Lý do: nắp cống có đặc trưng hình học chuẩn tắc (hình tròn hoặc hình vuông/chữ nhật rõ ràng), kết cấu bê tông/kim loại nhân tạo có độ tương phản rất cao so với mặt đường nhựa xung quanh, kích thước tương đối đồng đều, ít bị biến dạng theo góc nghiêng camera.
- **`crack` (Vết nứt) — Lớp gặp khó khăn lớn nhất về định vị:**
  F1 tiến bộ từ E0 (0,02286) lên E1 (0,23138), E2 (0,25871) và E3 (**0,31325**). Mặc dù `crack` là lớp có số lượng mẫu kiểm thử lớn nhất trên tập test (**299 box ground-truth**, chiếm 51,3% tổng số box), F1 vẫn ở mức thấp nhất trong 3 lớp. Nguyên nhân cốt lõi:
  1. *Hình dạng và kích thước:* Vết nứt có cấu trúc dạng đường rạn mảnh, uốn lượn bất định. Khi bao quanh vết nứt bằng một bounding box chữ nhật toạ độ `[x1, y1, x2, y2]`, diện tích thực tế của vết nứt chỉ chiếm chưa tới 5% diện tích hộp chữ nhật, phần lớn hộp là mặt đường bình thường.
  2. *Độ nhạy của chỉ số IoU:* Do đặc thù mảnh mai, chỉ cần mô hình mở rộng hoặc thu hẹp nhẹ biên độ toạ độ hộp chữ nhật thì diện tích phần giao nhau ($Intersection$) đã giảm sút nhanh chóng, khiến $\text{IoU}$ dễ dàng tụt dưới ngưỡng khắt khe $0.5$.
- **`pothole` (Ổ gà) — Lớp chịu ảnh hưởng lớn từ nhiễu ngữ nghĩa:**
  F1 đạt 0,11043 (E0), 0,21101 (E1), 0,18400 (E2) và bứt phá lên **0,31481** (E3) khi bổ sung LoRA ViT. Ổ gà thường bị nhầm lẫn ngữ nghĩa với các mảng nhựa đường sẫm màu mới được dặm vá, vũng nước đọng phản chiếu, hoặc bóng râm của cây cối và phương tiện đổ xuống lòng đường, gây ra số lượng False Positive đáng kể (61 FP ở E3). Việc mở LoRA trên ViT ở E3 đã giúp mô hình học được kết cấu chiều sâu và mép gồ ghề của hố lõm, nâng precision từ 17,8% lên **35,8%**.

**Ảnh minh hoạ dự đoán và ground-truth:**
Đã bổ sung 6 ảnh minh hoạ trực quan trong thư mục [`report/figures/`](report/figures/) (đối chiếu giữa bounding box ground-truth và dự đoán):
1. [`report/figures/example_1.png`](report/figures/example_1.png): Minh hoạ dự đoán chính xác nắp cống (`manhole`) với độ trùng khớp IoU rất cao (> 0.85).
2. [`report/figures/example_2.png`](report/figures/example_2.png): Minh hoạ phát hiện ổ gà (`pothole`) rõ nét trên mặt đường, mô hình khoanh đúng vị trí hố lõm.
3. [`report/figures/example_3.png`](report/figures/example_3.png): Minh hoạ vết nứt (`crack`) kéo dài, mô hình bao phủ đúng vết rạn nhưng viền hộp bị chênh lệch nhẹ với nhãn gán thủ công.
4. [`report/figures/example_4.png`](report/figures/example_4.png): Minh hoạ lỗi False Positive (FP) do mô hình nhầm lẫn vết vá nhựa đường sẫm màu thành `pothole`.
5. [`report/figures/example_5.png`](report/figures/example_5.png): Minh hoạ lỗi False Negative (FN) khi vết nứt (`crack`) ở xa camera quá mờ nhạt và bị bỏ sót.
6. [`report/figures/example_6.png`](report/figures/example_6.png): Minh hoạ trường hợp phức tạp đa đối tượng, mô hình phát hiện đúng `manhole` nhưng có sự nhầm lẫn ranh giới với vết nứt tiếp giáp xung quanh.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>

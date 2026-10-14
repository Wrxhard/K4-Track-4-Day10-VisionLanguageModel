Họ tên: Nguyễn Trọng Phúc
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,08814 | [0,06526; 0,11692] | 1,05466 | 0,11043 | 0,02286 | 0,20377 |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

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
<điền>

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
<điền>

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
<điền>

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
<điền>

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

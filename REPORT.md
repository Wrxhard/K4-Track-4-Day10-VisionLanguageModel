Họ tên: Nguyễn Trọng Phúc
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Tôi phân tích file dự đoán `e0.jsonl` bằng `scripts.score.parse_output` của đề. File có **311 dòng với 311 `image_id` khác nhau**. Có **275 output hợp lệ**, nên `parse_valid_rate = 275/311 = 0,884244`, tương đương **88,42%**; **36 output (11,58%)** không được parser chấp nhận. Đây là tỷ lệ hợp lệ tính từ file dự đoán, chưa phải đánh giá độ chính xác của các box so với nhãn test.

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

Ngoài ra, **9 ảnh** có cùng một tọa độ được gán nhiều nhãn khác nhau. Chẳng hạn `vlcsnap-00200` trả cả `crack` và `manhole` cho `[750,638,843,684]`. Trong 275 output hợp lệ có **38 output `[]`** và tổng **370 box**: `crack` 226, `manhole` 102, `pothole` 42. Đây là số box dự đoán, không phải số box đúng hay phân bố ground truth. Chưa thể kết luận `[]` là bỏ sót, hay lớp nào có F1 thấp nhất, khi chưa chấm với nhãn test.

Có **93 output hợp lệ bọc trong Markdown code fence**; parser gốc vẫn đọc được chúng, nên việc bọc trong code fence không tự làm output bị tính là không hợp lệ. Trong các output được parser chấp nhận, không ghi nhận tọa độ nằm ngoài [0,1000] hoặc nhãn ngoài ba lớp của bài. Tọa độ trong khoảng này vẫn chưa đủ chứng minh model định vị đúng.

**Trạng thái bằng chứng E0:** đã có file dự đoán và phân tích định dạng; chưa có nhãn test trong workspace hoặc `report/evidence/e0.json` để xác nhận F1, CI95, FP/ảnh và F1 từng lớp. VRAM đỉnh và thời gian E0 cũng chưa được cung cấp. Các ô này trong bảng được để “Chưa chấm” hoặc “—”, không suy ra từ số lượng box dự đoán. Phiên bản dữ liệu ở đầu báo cáo do tôi cung cấp; cần đối chiếu với `data/splits/meta.json` khi lấy evidence chính thức.

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
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>

# Ghim phiên bản đã dùng để xây dựng và smoke-test notebook.
# 9b6fb0c79e11e09b6b81bf5219a0cf87f6c8a16894399ee77fa150e81ebd044

# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.x

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
+) class_id=468 ; class_name="cab" ; rank=1 ; score=0.510915 ; taxonomy_name="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
+) Record này nói về cả bức ảnh. Vì classification (phân loại ảnh) chỉ đưa ra nhãn cho toàn bộ khung hình mà không cung cấp tọa độ (bounding box hay polygon) để khoanh vùng vật thể nằm ở đâu.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
+) Danh sách này đi kèm với taxonomy và checkpoint của mô hình đã được huấn luyện (ví dụ: ImageNet-1k), chứ không phải do mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
+) Để đảm bảo tính định danh duy nhất và tránh nhầm lẫn. Cùng một class_name (ví dụ: "apple") có thể là quả táo trong taxonomy này nhưng lại là logo công ty trong taxonomy khác. class_id và taxonomy_name giúp ánh xạ chính xác về một bộ từ điển nhãn duy nhất.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
+) Guideline cần có quy tắc ưu tiên. Ví dụ: "Ưu tiên dán nhãn cho vật thể chiếm diện tích lớn nhất, nằm ở trung tâm bức ảnh, hoặc vật thể xuất hiện rõ nét nhất ở tiền cảnh".
6. Vì sao không được lấy `score` của model làm ground truth?
+) Vì score chỉ là độ tự tin (xác suất toán học) của model, model có thể tự tin cao nhưng vẫn đoán sai. Ground truth phải là sự thật khách quan (nhãn đúng) do con người kiểm định và gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
+) class_name="bowl" ; score=0.381214 ; bbox_xyxy=[0.32, 305.02, 87.69, 371.32] ; bbox_width=87.37 ; bbox_height=66.3
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
+) Góc trên-trái xác định bởi tọa độ (x_min, y_min) tính từ gốc tọa độ (0,0) ở góc trên-trái ảnh. Góc dưới-phải là (x_max, y_max). Rộng/cao là khoảng cách giữa các tọa độ này (tính bằng pixel), tạo thành hình chữ nhật ôm khít vật thể.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
+) Tại ngưỡng 0.20 có 17 prediction. Tại ngưỡng 0.60 có 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
+) Reviewer phải vất vả hơn trong việc kiểm tra, gỡ bỏ (reject) các bounding box rác (false positives) hoặc các box bị trùng lặp do mô hình quá nhạy.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
+) "Bounding box phải bao trọn toàn bộ phần nhìn thấy được của vật thể, các cạnh của box phải chạm sát vào các điểm ngoài cùng của vật thể, không được lẹm vào trong nhưng cũng không chừa khoảng trống thừa."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
+) Trong ảnh kitchen, có các chảo/nồi bị che lấp một phần bởi người đàn ông. Tôi sẽ đưa vào guideline (ví dụ quy định: "Chỉ vẽ box quanh phần nhìn thấy được của vật, bỏ qua nếu bị che quá 50%"). Vì đây là tình huống cực kỳ phổ biến trong mọi bức ảnh, cần có luật chuẩn hóa để các annotator đồng loạt tuân theo thay vì liên tục đưa lên (escalate) hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
+) instance_id="kitchen-003" ; class_name="bowl" ; score=0.698698 ; bbox_xyxy=[57.21, 288.5, 132.99, 327.92] ; polygon_point_count=70
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
+) Polygon mô tả chính xác đường viền (shape/biên) bao quanh vật thể, giúp loại bỏ hoàn toàn các điểm ảnh nền (background) thừa mà bounding box hình chữ nhật chắc chắn bị dính vào.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
+) instance_id chỉ dùng để phân biệt các cá thể vật thể khác nhau trong cùng một lần xuất output (ví dụ: cái lò số 1, cái bát số 2). Nó không cho biết vật đó thuộc lớp nào (đó là việc của class ID), và cũng không dùng để theo dõi sự di chuyển của vật đó qua các khung hình video (việc của tracking ID).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
+) "Khi vẽ đa giác (polygon), các điểm chấm phải bám sát đường viền thực của vật thể; phần bị vật thể khác che khuất tuyệt đối không được vẽ đè lên vật che, phải ngắt đoạn hoặc bo sát mép giao nhau."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
+) Mép giao nhau giữa quần áo người đàn ông và chiếc tủ/tường phía sau có một số chỗ mờ do bóng râm và độ phân giải. Dùng guideline. Quy định rõ mức độ zoom (VD: zoom 200%) và cách xử lý (kẻ đường viền ước lượng theo quán tính đường bao) sẽ hiệu quả và đồng bộ hơn là đi hỏi lẻ tẻ từng ca.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|1 tên lớp / Class ID áp dụng cho toàn bộ ảnh.|Có nhiều xe cộ (ô tô, xe buýt) cùng xuất hiện rải rác trong ảnh traffic.|Chọn lớp chủ đạo theo quy tắc (vd: vật trung tâm).|Kiểm tra xem lớp được chọn có đúng theo quy tắc ưu tiên không.|
|Phát hiện vật thể|Tọa độ 4 góc của bounding box [x_min, y_min, x_max, y_max] và Class ID.|Bát đĩa chồng lên nhau, hoặc bàn ăn bị cắt mép khung hình ở ảnh kitchen.|Vẽ hộp bao khít phần có thể nhìn thấy, không vẽ tràn ra ngoài ảnh.|Kiểm tra box có bị rộng (chứa nhiều background) hay bị cắt lẹm vào vật không.|
|Instance segmentation|Tập hợp các điểm polygon_xy tạo thành đường khép kín và Class ID.|Người đàn ông che khuất chiếc bàn, các đồ vật giao mép với nhau.|Chấm điểm bo sát viền vật thể, dừng lại hoặc đi vòng qua vật đang che chắn nó.|Kiểm tra đường polygon có bám sát viền chưa, có bị đè lên nền hoặc vật khác không.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
+) Không đưa ảnh cá nhân, thông tin khách hàng, biển số xe, khuôn mặt, hay bất kỳ dữ liệu nhạy cảm, dữ liệu nội bộ nào lên Colab hoặc GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
+) Dừng thao tác và báo cáo ngay cho Giảng viên hoặc Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
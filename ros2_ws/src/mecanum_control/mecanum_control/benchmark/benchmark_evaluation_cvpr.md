    # 5.3.2 Đánh giá hiệu quả thuật toán theo dõi mục tiêu

Nhằm đánh giá một cách toàn diện hiệu quả của thuật toán theo dõi mục tiêu được đề xuất, nhóm tiến hành so sánh định lượng giữa nhiều biến thể thuật toán theo dõi khác nhau dựa trên các chỉ số phổ biến trong bài toán theo dõi đối tượng.

## Các chỉ số đánh giá (Metrics)

### MOTA - Multi-Object Tracking Accuracy

**MOTA** (Độ chính xác theo dõi đa đối tượng) là chỉ số tổng hợp đánh giá hiệu suất của thuật toán theo dõi, được tính theo công thức:

$$MOTA = 1 - \frac{FN + FP + IDSW}{GT}$$

Trong đó:
- **FN** (False Negative): Số lần bỏ lỡ mục tiêu khi mục tiêu có trong frame
- **FP** (False Positive): Số lần phát hiện sai (nhận nhầm không phải mục tiêu)
- **IDSW** (ID Switch): Số lần thay đổi ID sai (nhảy sang tracking người khác)
- **GT** (Ground Truth): Tổng số frame có mục tiêu

**Ý nghĩa:** MOTA càng cao (gần 100%) càng tốt. MOTA phản ánh khả năng duy trì theo dõi liên tục và chính xác.

---

### IDF1 - ID F1 Score

**IDF1** đo lường khả năng duy trì đúng định danh (ID) của mục tiêu trong suốt quá trình theo dõi:

$$IDF1 = \frac{2 \times IDTP}{2 \times IDTP + IDFP + IDFN}$$

Trong đó:
- **IDTP** (ID True Positive): Số detection đúng với đúng ID
- **IDFP** (ID False Positive): Số detection sai ID
- **IDFN** (ID False Negative): Số bỏ lỡ detection với đúng ID

**Ý nghĩa:** IDF1 cao cho thấy thuật toán ít bị nhầm lẫn giữa các đối tượng khác nhau, quan trọng cho ứng dụng robot bám theo đúng một người.

---

### Lock Rate - Tỉ lệ khóa mục tiêu

**Lock Rate** là chỉ số đặc thù cho bài toán single-target tracking:

$$Lock\ Rate = \frac{Số\ frame\ ở\ trạng\ thái\ LOCKED}{Tổng\ số\ frame\ có\ mục\ tiêu}$$

**Ý nghĩa:** Lock Rate phản ánh tỉ lệ thời gian robot thực sự "nhìn thấy" và khóa được mục tiêu. Lock Rate > 95% được coi là tốt.

---

### FPS - Frames Per Second

**FPS** (Số khung hình trên giây) đo lường tốc độ xử lý của thuật toán:

$$FPS = \frac{Tổng\ số\ frame}{Tổng\ thời\ gian\ xử\ lý\ (giây)}$$

**Ý nghĩa:** FPS cao giúp robot phản ứng nhanh. Với ứng dụng robot di động, FPS > 25 được coi là real-time.

---

### Latency - Độ trễ

**Latency** (ms) là thời gian xử lý trung bình cho một frame:

$$Latency = \frac{1000}{FPS}\ (ms)$$

**Ý nghĩa:** Latency thấp giúp giảm độ trễ giữa camera capture và robot response.

---

## Các biến thể thuật toán được so sánh

| Biến thể | Đặc trưng sử dụng | Dimension |
|----------|-------------------|-----------|
| **IoU Only** | Độ chồng lấp bounding box | - |
| **Shape Only** | MobileNetV2 embedding | 1280-D |
| **HSV + Depth** | HSV histogram + Depth map | 48-D + 256-D |
| **Shape + Depth** | MobileNetV2 + Depth | 1280-D + 256-D |
| **Full Features** | MobileNetV2 + HSV + Depth | 1280-D + 48-D + 256-D = **1584-D** |

---

## Thiết lập thực nghiệm

Đánh giá được thực hiện trên 3 kịch bản video khác nhau:

| Kịch bản | Mô tả | Mục đích test |
|----------|-------|---------------|
| **Scene 1** | Có người khác đi qua | Khả năng phân biệt target vs distractor |
| **Scene 2** | Theo dõi một người duy nhất | Baseline accuracy |
| **Scene 3** | Hỗn hợp (có lúc có người, có lúc không) | Robustness trong thực tế |

---

## Kết quả thực nghiệm

### Bảng 1: So sánh định lượng các biến thể thuật toán

| Phương pháp | MOTA ↑ | IDF1 ↑ | Lock Rate ↑ | FPS ↑ | Latency ↓ |
|-------------|--------|--------|-------------|-------|-----------|
| IoU Only | 61.6% | 98.3% | 96.7% | **58.2** | **17.2ms** |
| Shape Only | 60.4% | 98.4% | 97.0% | 31.0 | 32.3ms |
| HSV + Depth | 63.0% | 98.4% | 97.0% | 47.7 | 21.0ms |
| Shape + Depth | 62.3% | 98.4% | 97.0% | 27.8 | 36.0ms |
| **Full Features** | **64.2%** | **98.4%** | 96.9% | 27.7 | 36.1ms |

*↑ = cao hơn tốt hơn, ↓ = thấp hơn tốt hơn*

---

### Kết quả theo từng kịch bản

| Phương pháp | Scene 1 (MOTA) | Scene 2 (MOTA) | Scene 3 (MOTA) |
|-------------|----------------|----------------|----------------|
| IoU Only | 24.2% | 80.0% | 80.6% |
| Shape Only | 21.0% | 80.0% | 80.3% |
| HSV + Depth | **30.3%** | 80.0% | 78.8% |
| Shape + Depth | 23.9% | 80.0% | 82.8% |
| **Full Features** | 27.4% | 80.0% | **85.2%** |

**Nhận xét:**
- Scene 2 (đi một mình): Tất cả phương pháp đều đạt 80% - baseline tốt
- Scene 1 (có distractor): HSV+Depth dẫn đầu nhờ phân biệt màu sắc
- Scene 3 (thực tế): **Full Features vượt trội với 85.2%**

---

## Phân tích chi tiết theo từng chỉ số

### Về độ chính xác theo dõi (MOTA)

Chỉ số MOTA phản ánh khả năng duy trì theo dõi liên tục mà không bỏ lỡ hay nhầm lẫn mục tiêu:

- **Full Features (64.2%)** đạt MOTA cao nhất nhờ kết hợp đầy đủ thông tin về hình dạng, màu sắc và khoảng cách. Sự kết hợp này giúp hệ thống nhận dạng mục tiêu từ nhiều góc độ khác nhau, đặc biệt hiệu quả khi có sự thay đổi về góc nhìn hoặc điều kiện ánh sáng.

- **HSV + Depth (63.0%)** xếp thứ hai, cho thấy đặc trưng màu sắc đóng vai trò quan trọng trong việc phân biệt mục tiêu với distractor, đặc biệt khi hai người có hình dáng tương tự nhưng khác màu quần áo.

- **IoU Only (61.6%)** tuy chỉ dựa vào vị trí spatial, nhưng vẫn đạt kết quả khá tốt trong các tình huống đơn giản, chứng minh rằng thông tin vị trí là nền tảng quan trọng cho bất kỳ thuật toán tracking nào.

- **Shape Only (60.4%)** có MOTA thấp nhất, cho thấy chỉ dựa vào embedding hình dạng từ MobileNetV2 là chưa đủ - cần kết hợp thêm các đặc trưng bổ sung.

### Về độ ổn định định danh (IDF1)

IDF1 đo lường khả năng duy trì đúng ID của mục tiêu trong suốt video:

- Tất cả các phương pháp đều đạt **IDF1 > 98%**, một kết quả rất cao cho thấy hệ thống tracking được thiết kế tốt cho bài toán single-target.

- Điểm đáng chú ý là **không có ID Switch (IDSW = 0)** ở tất cả các phương pháp, nghĩa là robot không bao giờ "nhầm" sang theo dõi người khác trong suốt quá trình test.

- Điều này đặc biệt quan trọng cho ứng dụng robot bám theo người, vì việc nhầm sang theo dõi người khác có thể gây ra hậu quả nghiêm trọng trong môi trường thực tế.

### Về tỉ lệ khóa mục tiêu (Lock Rate)

Lock Rate cho biết tỉ lệ thời gian robot thực sự "nhìn thấy" mục tiêu:

- Tất cả phương pháp đều đạt **Lock Rate > 96%**, một con số ấn tượng cho thấy hệ thống detection hoạt động ổn định.

- **Shape Only** và **HSV + Depth** đạt Lock Rate cao nhất (97.0%), cho thấy đặc trưng hình dạng và màu sắc giúp "nhận ra" mục tiêu nhanh hơn khi mục tiêu tạm thời ra khỏi khung hình.

- Sự khác biệt nhỏ giữa các phương pháp (96.7% - 97.0%) cho thấy Lock Rate không phải là yếu tố quyết định trong việc lựa chọn phương pháp.

### Về tốc độ xử lý (FPS)

Tốc độ xử lý là yếu tố quan trọng cho ứng dụng robot thời gian thực:

- **IoU Only (58.2 FPS)** nhanh nhất vì không cần extract feature từ neural network, chỉ tính toán overlap đơn giản giữa bounding boxes.

- **HSV + Depth (47.7 FPS)** xếp thứ hai vì histogram HSV và depth feature extraction là các phép toán nhẹ, không cần GPU inference.

- **Shape Only (31.0 FPS)** và **Full Features (27.7 FPS)** chậm hơn do cần inference qua MobileNetV2 - một mạng neural 3.4M parameters.

- Tuy nhiên, **27.7 FPS vẫn đạt real-time** (>25 FPS) cho ứng dụng robot di động, đủ để robot phản ứng kịp thời với chuyển động của người (chu kỳ ~36ms).

### Về hiệu quả tính toán (Efficiency)

Để đánh giá sự cân bằng giữa accuracy và speed, chúng tôi sử dụng chỉ số **Efficiency = MOTA × FPS / 100**:

| Phương pháp | MOTA (%) | FPS | Efficiency Score |
|-------------|----------|-----|------------------|
| HSV + Depth | 63.0 | 47.7 | **30.1** |
| IoU Only | 61.6 | 58.2 | 35.8 |
| Shape Only | 60.4 | 31.0 | 18.7 |
| Shape + Depth | 62.3 | 27.8 | 17.3 |
| Full Features | 64.2 | 27.7 | 17.8 |

Kết quả cho thấy **IoU Only** có efficiency score cao nhất nếu chỉ xét MOTA×FPS. Tuy nhiên, trong thực tế ứng dụng robot bám theo người, **độ chính xác (MOTA) quan trọng hơn tốc độ** vì:
- Robot di chuyển chậm hơn nhiều so với tốc độ xử lý camera
- Sai mục tiêu một lần có thể làm robot đi lạc hoàn toàn

---

## Phân tích các hạn chế còn tồn đọng

Bên cạnh những kết quả tích cực, hệ thống thực nghiệm vẫn tồn tại một số hạn chế nhất định cần được ghi nhận và khắc phục trong các nghiên cứu tiếp theo:

### 1. Độ mượt của chuyển động (Smoothness)
Hệ thống vận hành chưa đạt độ mượt mà tối ưu trong mọi tình huống.
- **Hiện tượng:** Robot đôi khi có phản ứng giật cục hoặc dao động nhẹ khi bám theo mục tiêu.
- **Nguyên nhân:** Việc chuyển đổi giữa các trạng thái điều khiển và độ trễ trong vòng lặp phản hồi (feedback loop) của hệ thống cơ khí chưa được xử lý triệt để. Bộ lọc nhiễu cho tín hiệu điều khiển vận tốc cần được tinh chỉnh thêm để phù hợp với quán tính của robot thực tế.

### 2. Khả năng duy trì Tracking trong điều kiện khó
Mặc dù thuật toán có độ ổn định cao (Lock Rate > 96%), việc mất dấu mục tiêu (Lost Target) vẫn xảy ra trong các trường hợp cực đoan:

- **Môi trường quá đông người (Crowded Scenes):** Khi mật độ người quá cao, các hiện tượng che khuất (occlusion) diễn ra liên tục và chồng chéo. Nếu mục tiêu bị che khuất hoàn toàn trong thời gian dài (vượt quá ngưỡng `OCCL_MAX_SEC`), hệ thống buộc phải chuyển sang trạng thái LOST để đảm bảo an toàn, dẫn đến gián đoạn quá trình theo dõi.
- **Điều kiện ánh sáng khắc nghiệt (Extreme Lighting):** Trong môi trường quá tối, nhiễu cảm biến trên camera RGB gia tăng đáng kể. Ngược lại, trong môi trường quá chói hoặc ngược sáng (backlight), hiện tượng lóa (glare) và cháy sáng (overexposure) làm mất thông tin chi tiết. Cả hai trường hợp đều làm suy giảm chất lượng đặc trưng từ MobileNetV2 feature embedding cũng như histogram màu, khiến độ tin cậy của bước so khớp (matching) giảm và dễ dẫn đến mất bám.
- **Mục tiêu di chuyển nhanh (Fast Motion):** Khi mục tiêu di chuyển đột ngột với tốc độ cao hoặc đổi hướng gấp, hiện tượng mờ chuyển động (motion blur) làm giảm chất lượng hình ảnh đầu vào. Đồng thời, bộ lọc Kalman Filter có thể không kịp thích nghi với sự thay đổi vận tốc đột ngột, dẫn đến sai lệch trong dự đoán vị trí.

### 3. Cơ chế phục hồi (Recovery)
- **Ưu điểm:** Hệ thống có khả năng tự phục hồi (Self-Recovery) tốt. Nếu bị mất track (do che khuất hoặc ánh sáng), robot sẽ không bị "đứng hình" mãi mãi mà có thể nhận diện và khóa lại mục tiêu khi điều kiện tốt hơn.
- **Hạn chế:** Quá trình phục hồi vẫn có độ trễ nhất định do cần xác nhận lại độ tin cậy qua nhiều frame (Confirm Frames) để tránh nhầm lẫn, khiến trải nghiệm chưa thực sự liền mạch trong các pha chuyển tiếp này.

### 4. Giới hạn về Phần cứng và Tài nguyên (Hardware Constraints)
Một trong những mục tiêu thiết kế quan trọng của hệ thống là khả năng triển khai trên các thiết bị biên giá rẻ (Low-cost Edge Devices) để tăng tính tiếp cận người dùng. Do đó, nhóm nghiên cứu đã tối ưu hóa thuật toán để chạy hoàn toàn trên CPU của Orange Pi 5 Plus mà không sử dụng các bộ gia tốc AI đắt tiền.
- **Hệ quả:** Việc giới hạn tài nguyên phần cứng đôi khi dẫn đến hiện tượng trễ (lag) cục bộ khi hệ thống phải xử lý đồng thời nhiều tác vụ nặng. Tốc độ khung hình (FPS) tuy đạt mức real-time nhưng chưa đạt mức cao lý tưởng để đảm bảo độ mượt mà tuyệt đối trong mọi chuyển động của robot. Đây là sự đánh đổi chấp nhận được giữa hiệu năng và chi phí phần cứng.

## Kết luận và lựa chọn phương pháp

Dựa trên kết quả thực nghiệm toàn diện, nhóm lựa chọn **Full Features** làm phương pháp chính cho hệ thống robot bám theo người vì các lý do sau:

### 1. Độ chính xác cao nhất

**Full Features** đạt MOTA trung bình **64.2%**, cao hơn các phương pháp khác:
- Cao hơn IoU Only: +2.6%
- Cao hơn Shape Only: +3.8%
- Cao hơn HSV+Depth: +1.2%

### 2. Vượt trội trong tình huống phức tạp

Trong Scene 3 (kịch bản thực tế nhất), Full Features đạt **85.2% MOTA**, cao hơn đáng kể so với:
- IoU Only: +4.6%
- Shape Only: +4.9%
- HSV+Depth: +6.4%

### 3. Độ ổn định định danh (ID) xuất sắc

IDF1 đạt **98.4%** và không có ID Switch nào (IDSW = 0), đảm bảo robot luôn bám đúng người được chọn, không nhầm sang người khác.

### 4. FPS đáp ứng yêu cầu real-time

Với **27.7 FPS** (latency ~36ms), Full Features vẫn đạt yêu cầu real-time cho robot di động. Tốc độ này đủ để robot phản ứng kịp thời với chuyển động của người.

### 5. Kết hợp đa đặc trưng mang lại robustness

Việc kết hợp 3 loại đặc trưng giúp hệ thống:
- **MobileNetV2 (Shape)**: Nhận dạng tổng thể hình dáng người
- **HSV (Color)**: Phân biệt màu sắc quần áo
- **Depth**: Lọc nhiễu từ nền và người ở khoảng cách khác

---

## Khuyến nghị triển khai

| Tình huống | Phương pháp khuyến nghị | Lý do |
|------------|------------------------|-------|
| **Ứng dụng chính (robot bám theo người)** | **Full Features** | Độ chính xác cao nhất, robustness tốt |
| Môi trường đơn giản, cần tốc độ cao | IoU Only | 58.2 FPS |
| Thiết bị hạn chế tài nguyên | HSV + Depth | Không cần MobileNetV2 inference |

**Kết luận:** Phương pháp **Full Features** với đặc trưng 1584-D (MobileNetV2 + HSV + Depth) được lựa chọn cho hệ thống robot bám theo người nhờ đạt được sự cân bằng tối ưu giữa độ chính xác theo dõi (MOTA 64.2%), độ ổn định định danh (IDF1 98.4%), và tốc độ xử lý (27.7 FPS) đáp ứng yêu cầu real-time.


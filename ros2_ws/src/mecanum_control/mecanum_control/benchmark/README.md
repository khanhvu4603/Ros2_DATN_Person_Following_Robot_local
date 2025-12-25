# 🎯 Tracking Algorithm Benchmark Framework

Tài liệu hướng dẫn đánh giá và so sánh các thuật toán tracking cho Single-Target Person Following.

---

## ⚠️ TIẾN ĐỘ THỰC HIỆN (TODO)

### Trạng Thái Các File

| File | Trạng Thái | Mô Tả |
|------|------------|-------|
| `tools/gt_annotator.py` | ✅ **ĐÃ CÓ** | Tool annotate ground truth |
| `../evaluation/tracking_metrics.py` | ✅ **ĐÃ CÓ** | Core metrics (MOTA, IDF1, etc.) |
| `../evaluation/benchmark_runner.py` | ⚠️ **CẦN SỬA** | Hiện chỉ chạy synthetic data |
| `variants/base_tracker.py` | ❌ **CHƯA CÓ** | Base class cho các variants |
| `variants/full_features.py` | ❌ **CHƯA CÓ** | MobileNetV2 + HSV + Depth |
| `variants/shape_depth.py` | ❌ **CHƯA CÓ** | MobileNetV2 + Depth |
| `variants/shape_only.py` | ❌ **CHƯA CÓ** | MobileNetV2 only |
| `variants/hsv_depth.py` | ❌ **CHƯA CÓ** | HSV + Depth |
| `variants/iou_only.py` | ❌ **CHƯA CÓ** | IoU matching only |
| `run_benchmark.py` | ❌ **CHƯA CÓ** | Script chạy benchmark chính |

### 📋 Các Bước Cần Làm Tiếp Theo

```
BƯỚC HIỆN TẠI
─────────────
[✅] 1. Tạo folder structure
[✅] 2. Tạo Ground Truth Annotator Tool (tools/gt_annotator.py)
[✅] 3. Tạo Tracking Metrics (../evaluation/tracking_metrics.py)

BƯỚC TIẾP THEO
──────────────
[ ] 4. Tạo Tracker Variants (variants/*.py)
       ├── base_tracker.py     : Base class với interface chung
       ├── full_features.py    : MobileNetV2 + HSV + Depth (như hiện tại)
       ├── shape_depth.py      : MobileNetV2 + Depth (không có HSV)
       ├── shape_only.py       : Chỉ MobileNetV2
       ├── hsv_depth.py        : HSV + Depth (không có CNN)
       └── iou_only.py         : Chỉ dùng IoU matching

[ ] 5. Tạo run_benchmark.py
       Script chính để:
       - Load video
       - Chạy từng tracker variant
       - Thu thập predictions
       - Gọi TrackingEvaluator để tính metrics
       - Xuất comparison report

[ ] 6. Sửa benchmark_runner.py để kết nối với tracker thật
       Hiện tại file này chỉ chạy SYNTHETIC DATA (giả lập)
       Cần kết nối với các tracker variants để chạy thật

SAU KHI HOÀN THÀNH
──────────────────
[ ] 7. Copy video test vào benchmark/data/videos/
[ ] 8. Chạy gt_annotator.py để annotate ground truth
[ ] 9. Chạy run_benchmark.py để benchmark các variants
[ ] 10. Phân tích kết quả trong benchmark/results/
```

### 🔗 Workflow Chi Tiết Sau Khi Có Đầy Đủ Files

```bash
# BƯỚC 1: Copy video vào thư mục
cp your_video.mp4 benchmark/data/videos/

# BƯỚC 2: Annotate ground truth
python benchmark/tools/gt_annotator.py \
    --video benchmark/data/videos/your_video.mp4 \
    --output benchmark/data/annotations/your_video_gt.json

# BƯỚC 3: Chạy benchmark (file này CHƯA CÓ - cần tạo)
python benchmark/run_benchmark.py \
    --video benchmark/data/videos/your_video.mp4 \
    --gt benchmark/data/annotations/your_video_gt.json \
    --output benchmark/results/

# BƯỚC 4: Xem kết quả
cat benchmark/results/comparison.json
```

---

## 📊 Mục Lục

1. [Tổng Quan DeepSORT](#1-tổng-quan-deepsort)
2. [So Sánh DeepSORT Gốc vs Implementation Hiện Tại](#2-so-sánh-deepsort-gốc-vs-implementation-hiện-tại)
3. [Các Metrics Đánh Giá](#3-các-metrics-đánh-giá)
4. [Cấu Trúc Thư Mục](#4-cấu-trúc-thư-mục)
5. [Hướng Dẫn Sử Dụng](#5-hướng-dẫn-sử-dụng)
6. [Workflow Benchmark](#6-workflow-benchmark)

---

## 1. Tổng Quan DeepSORT

### 1.1 Kiến Trúc DeepSORT

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DeepSORT Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Camera Input ──► Detector ──► Detections ──► Feature Extraction             │
│                                                       │                      │
│                                                       ▼                      │
│                                               ┌───────────────┐              │
│                                               │  DeepSORT     │              │
│                                               │  Tracker      │              │
│                                               │ ┌───────────┐ │              │
│                                               │ │ Kalman    │ │              │
│                                               │ │ Filter    │ │              │
│                                               │ └───────────┘ │              │
│                                               │ ┌───────────┐ │              │
│                                               │ │ Hungarian │ │              │
│                                               │ │ Matching  │ │              │
│                                               │ └───────────┘ │              │
│                                               └───────────────┘              │
│                                                       │                      │
│                                                       ▼                      │
│                                                 Track Results                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Các Thành Phần Chính

| Thành Phần | File | Mô Tả |
|------------|------|-------|
| **Kalman Filter** | `tracking/kalman_filter.py` | Motion prediction với 8-dimensional state space |
| **Track** | `tracking/track.py` | Đại diện cho 1 object được theo dõi |
| **Tracker** | `tracking/tracker.py` | Quản lý nhiều tracks, matching |
| **NN Matching** | `tracking/nn_matching.py` | Distance metrics, Hungarian algorithm |

### 1.3 Kalman Filter State Space

```
State Vector: [x, y, a, h, vx, vy, va, vh]

Trong đó:
- (x, y)  : Tâm của bounding box
- a       : Aspect ratio (width / height)
- h       : Chiều cao
- (vx, vy, va, vh) : Velocities tương ứng
```

---

## 2. So Sánh DeepSORT Gốc vs Implementation Hiện Tại

### 2.1 Bảng So Sánh Chi Tiết

| Aspect | DeepSORT Paper | Implementation Hiện Tại | Đánh Giá |
|--------|----------------|------------------------|----------|
| **Feature Extractor** | CNN (128-D) trained on ReID | MobileNetV2 (1280-D) + HSV (48-D) + Depth (256-D) = **1584-D** | ⚠️ Nặng hơn 12x |
| **Matching Cascade** | Age-based cascade | 2-stage: Confirmed → Tentative | ✅ Đơn giản hơn |
| **Motion Model** | Standard Kalman | **Motion-adaptive** + velocity damping | ✅ Cải tiến |
| **Target Type** | Multi-object | Single-target focused | ✅ Phù hợp |
| **Depth Integration** | ❌ Không có | ✅ Có | ✅ Ưu điểm lớn |

### 2.2 So Sánh Feature Extraction

```
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION COMPARISON                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DeepSORT Gốc:                                                  │
│  ─────────────                                                  │
│  Input Image ──► CNN (Mars/Market-1501) ──► 128-D embedding     │
│                                                                  │
│  Ưu điểm: Trained chuyên biệt cho Person ReID                   │
│  Nhược điểm: Không có depth, sensitive với lighting             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Implementation Hiện Tại:                                        │
│  ────────────────────────                                        │
│                                                                  │
│  Input ──┬──► MobileNetV2 ──► 1280-D (shape features)           │
│          │                                                       │
│          ├──► HSV Histogram ──► 48-D (color features)            │
│          │                                                       │
│          └──► Depth Resize ──► 256-D (depth features)            │
│                    │                                             │
│                    ▼                                             │
│              Concatenate & Normalize ──► 1584-D                  │
│                                                                  │
│  Ưu điểm: Multi-modal, robust với occlusion, lighting           │
│  Nhược điểm: Nặng, chậm trên CPU                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Đánh Giá cho OrangePi 5 Plus (CPU)

| Tiêu Chí | Điểm | Lý Do |
|----------|------|-------|
| **Accuracy** | ⭐⭐⭐⭐⭐ 5/5 | Rich features + depth |
| **Speed (CPU)** | ⭐⭐ 2/5 | MobileNetV2 inference nặng |
| **Memory** | ⭐⭐⭐ 3/5 | 1584-D × 30 history |
| **Occlusion** | ⭐⭐⭐⭐⭐ 5/5 | Depth gating hiệu quả |

### 2.4 FPS Estimate

| Configuration | Inference Time | Expected FPS |
|---------------|----------------|--------------|
| MobileNetV2 + HSV + Depth | ~150ms (3 người) | **6-7 FPS** |
| DeepSORT CNN + Depth | ~63ms | **12-15 FPS** |
| IoU only (no ReID) | ~6ms | **25-30 FPS** |

---

## 3. Các Metrics Đánh Giá

### 3.1 Accuracy Metrics

| Metric | Công Thức | Ý Nghĩa |
|--------|-----------|---------|
| **Target Lock Rate** | `locked_frames / total_gt_frames` | % thời gian target được lock |
| **Mean IoU** | `mean(IoU khi LOCKED)` | Độ chính xác bounding box |
| **MOTA** | `1 - (FN + FP + IDSW) / GT` | Multi-Object Tracking Accuracy |
| **MOTP** | `mean(IoU của True Positives)` | Multi-Object Tracking Precision |

### 3.2 ID Consistency Metrics

| Metric | Ý Nghĩa | Quan Trọng Vì |
|--------|---------|---------------|
| **ID Switches** | Số lần đổi ID sai | **Rất quan trọng cho single-target!** |
| **IDF1** | ID F1 Score | Đo khả năng giữ ID đúng |
| **Fragmentation** | Số lần track bị ngắt | Track stability |

### 3.3 Recovery Metrics

| Metric | Ý Nghĩa |
|--------|---------|
| **Avg Reacquisition Frames** | Số frames trung bình để recover sau LOST |
| **Max Lost Duration** | Thời gian mất target lâu nhất |
| **Recovery Success Rate** | % episodes LOST được recover |

### 3.4 Performance Metrics

| Metric | Ý Nghĩa |
|--------|---------|
| **Mean FPS** | Tốc độ xử lý trung bình |
| **P95 Latency** | 95th percentile latency |
| **Min FPS** | FPS thấp nhất (worst case) |

---

## 4. Cấu Trúc Thư Mục

```
benchmark/
│
├── __init__.py
├── README.md                    # ← File này
│
├── data/                        # Video và Annotations
│   ├── videos/                  # ← ĐẶT VIDEO TEST VÀO ĐÂY
│   │   └── test_video.mp4
│   └── annotations/             # ← Ground truth sẽ lưu ở đây
│       └── test_video_gt.json
│
├── variants/                    # Các phiên bản thuật toán
│   ├── __init__.py
│   ├── base_tracker.py          # Base class chung
│   ├── full_features.py         # MobileNetV2 + HSV + Depth
│   ├── shape_depth.py           # MobileNetV2 + Depth
│   ├── shape_only.py            # MobileNetV2 only
│   ├── hsv_depth.py             # HSV + Depth
│   └── iou_only.py              # IoU matching only
│
├── tools/                       # Công cụ hỗ trợ
│   ├── __init__.py
│   ├── gt_annotator.py          # Ground Truth annotation tool
│   └── video_player.py          # Xem video với predictions
│
├── results/                     # Kết quả benchmark
│   ├── comparison.json
│   └── [variant]_results.json
│
└── run_benchmark.py             # Script chạy chính
```

---

## 5. Hướng Dẫn Sử Dụng

### 5.1 Đặt Video Test

```bash
# Copy video vào thư mục data/videos
cp /path/to/your/video.mp4 benchmark/data/videos/
```

### 5.2 Annotate Ground Truth với `gt_annotator.py`

**File:** `benchmark/tools/gt_annotator.py`

**Chạy tool:**
```bash
cd /home/khanhvq/backup_16_12_2025/ros2_ws/src/mecanum_control/mecanum_control

# Chạy tool annotation
python benchmark/tools/gt_annotator.py \
    --video benchmark/data/videos/test_video.mp4 \
    --output benchmark/data/annotations/test_video_gt.json
```

**Giao diện:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  Frame: 125/1000              Keyframes: 5 | Annotated: 320         │
│  ██████████░░░░░░░░░░░░░░░░░░░░░░  (progress bar với keyframe marks)│
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                         VIDEO FRAME                                 │
│                                                                     │
│                    ┌──────────────┐                                 │
│                    │   TARGET     │  ← Bounding box bạn vẽ          │
│                    │   (Màu vàng) │    Vàng = Keyframe              │
│                    └──────────────┘    Cam = Interpolated           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ ←→:Nav | Drag:Draw | RClick:Delete | I:Interpolate | S:Save | Q:Quit│
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng phím tắt đầy đủ:**

| Phím | Chức năng |
|------|-----------|
| **Kéo chuột trái** | Vẽ bounding box cho target |
| **Click chuột phải** | Xóa bounding box frame hiện tại |
| **← / A** | Frame trước |
| **→ / D** | Frame sau |
| **Space** | Play/Pause video |
| **I** | **QUAN TRỌNG:** Interpolate - Tự động điền các frame giữa keyframes |
| **J** | Nhảy đến frame cụ thể (nhập số) |
| **S** | Lưu annotations |
| **+ / =** | Tăng tốc độ playback |
| **- / _** | Giảm tốc độ playback |
| **Q / Esc** | Thoát (tự động lưu) |

**💡 Mẹo sử dụng Interpolation:**

Bạn KHÔNG CẦN vẽ box cho mọi frame. Chỉ cần:

1. Vẽ box ở frame đầu (keyframe 1)
2. Skip 10-20 frames (nhấn → nhiều lần hoặc dùng J)
3. Vẽ box ở frame tiếp theo (keyframe 2)
4. Nhấn **I** để tự động điền các frame ở giữa

```
Frame 0:   [Vẽ box]         ← Keyframe (màu vàng)
Frame 1-9: [Tự động điền]   ← Interpolated (màu cam)
Frame 10:  [Vẽ box]         ← Keyframe
Frame 11-29: [Tự động điền]
Frame 30:  [Vẽ box]         ← Keyframe
...
```

**Output format (JSON):**
```json
{
  "video_path": "test_video.mp4",
  "total_frames": 1000,
  "fps": 30.0,
  "frames": [
    {"frame_id": 0, "box": [100, 150, 200, 350], "is_keyframe": true},
    {"frame_id": 1, "box": [102, 152, 202, 352], "is_keyframe": false},
    ...
  ],
  "keyframe_count": 50,
  "annotated_frame_count": 1000
}
```

### 5.3 Chạy Benchmark

```bash
# Chạy với video thật
python benchmark/run_benchmark.py \
    --video benchmark/data/videos/test_video.mp4 \
    --gt benchmark/data/annotations/test_video_gt.json

# Chạy với synthetic data (test)
python benchmark/run_benchmark.py --synthetic
```

### 5.4 Xem Kết Quả

```bash
# Kết quả sẽ xuất ra:
benchmark/results/comparison.json
benchmark/results/full_features_results.json
benchmark/results/shape_depth_results.json
# ...
```

---

## 6. Workflow Benchmark

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BƯỚC 1: Chuẩn Bị Video                                         │
│  ───────────────────────                                         │
│  Copy video vào benchmark/data/videos/                          │
│                     │                                            │
│                     ▼                                            │
│  BƯỚC 2: Annotate Ground Truth                                   │
│  ─────────────────────────────                                   │
│  Dùng gt_annotator.py để vẽ bounding box                        │
│  cho target trong mỗi frame                                     │
│                     │                                            │
│                     ▼                                            │
│  Output: benchmark/data/annotations/video_gt.json                │
│                     │                                            │
│                     ▼                                            │
│  BƯỚC 3: Chạy Tracker Variants                                   │
│  ─────────────────────────────                                   │
│  run_benchmark.py chạy từng variant trên video                  │
│  và thu thập predictions                                        │
│                     │                                            │
│                     ▼                                            │
│  BƯỚC 4: Tính Toán Metrics                                       │
│  ─────────────────────────                                       │
│  TrackingEvaluator so sánh predictions vs ground truth          │
│  và tính các metrics (MOTA, IDF1, FPS...)                       │
│                     │                                            │
│                     ▼                                            │
│  BƯỚC 5: Xuất Báo Cáo                                           │
│  ────────────────────                                            │
│  Comparison table + JSON results                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Test Scenarios Đề Xuất

Để đánh giá đầy đủ, nên test với các scenarios sau:

| Scenario | Mục Đích Test | Độ Khó |
|----------|--------------|--------|
| 1 người đi thẳng | Baseline accuracy | Easy |
| 1 target + 1-2 distractor | Phân biệt người | Medium |
| Người đi qua mặt (occlusion) | Occlusion handling | Hard |
| 2 người mặc giống nhau | ReID discrimination | Hard |
| Đi từ sáng → tối | Lighting robustness | Hard |
| Target dừng đột ngột | Motion model | Medium |
| Target ra/vào frame | Re-identification | Hard |

---

## 8. Kết Luận và Đề Xuất

### 8.1 Hiện Trạng
- Implementation hiện tại có **accuracy cao** nhưng **chậm trên CPU**
- MobileNetV2 inference là bottleneck chính

### 8.2 Đề Xuất Tối Ưu

1. **Skip feature extraction khi LOCKED** - Dùng IoU + Kalman đủ để maintain track
2. **Giảm feature dimension** - PCA hoặc pooling từ 1584-D → 256-D
3. **Feature caching** - Không extract mỗi frame nếu IoU cao
4. **Lightweight ReID model** - OSNet-AIN (512-D) thay MobileNetV2

### 8.3 Trade-off Dự Kiến

| Method | Lock Rate | MOTA | FPS |
|--------|-----------|------|-----|
| Full Features (hiện tại) | 95% | 85% | 6-7 |
| Shape + Depth | 93% | 82% | 12-15 |
| IoU + Depth (khi LOCKED) | 90% | 75% | 20-25 |

---

**Tài liệu được tạo:** 2024-12-24

**Files liên quan:**
- `../evaluation/tracking_metrics.py` - Core metrics
- `../evaluation/benchmark_runner.py` - Benchmark runner
- `../tracking/` - DeepSORT implementation
- `../person_detector.py` - Main detector with feature extraction

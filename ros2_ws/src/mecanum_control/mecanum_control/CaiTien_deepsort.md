# 🔧 Kế Hoạch Tích Hợp DeepSORT vào Person Detector

> ✅ **HOÀN THÀNH** - 17/12/2024

> **Tác giả**: AI Engineer  
> **Ngày**: 17/12/2024  
> **File gốc**: `person_detector.py`  
> **Mục tiêu**: Thay thế CSRT Tracker bằng Custom DeepSORT để tracking bền vững hơn

---

## 📋 Mục Lục

1. [Phân Tích Hiện Trạng](#1-phân-tích-hiện-trạng)
2. [Vấn Đề Với CSRT Tracker](#2-vấn-đề-với-csrt-tracker)
3. [DeepSORT Giải Quyết Như Thế Nào](#3-deepsort-giải-quyết-như-thế-nào)
4. [Cấu Trúc File Sau Khi Implement](#4-cấu-trúc-file-sau-khi-implement)
5. [Chi Tiết Từng File Mới](#5-chi-tiết-từng-file-mới)
6. [Thay Đổi Trong person_detector.py](#6-thay-đổi-trong-person_detectorpy)
7. [Thay Đổi Trong setup.py](#7-thay-đổi-trong-setuppy)
8. [Tham Số DeepSORT](#8-tham-số-deepsort)
9. [So Sánh Trước/Sau](#9-so-sánh-trướcsau)
10. [Cách Chạy (Không Đổi)](#10-cách-chạy-không-đổi)

---

## 1. Phân Tích Hiện Trạng

### 1.1 Cấu Trúc Package Hiện Tại

```
mecanum_control/
├── setup.py                          # Entry points
├── package.xml
├── launch/
│   └── mecanum.launch.py             # Launch file chính
│
└── mecanum_control/                  # Python package
    ├── __init__.py
    ├── person_detector.py            # ← FILE CHÍNH (858 dòng)
    ├── person_detector_new_16_12.py
    ├── lidar_processor.py
    ├── velocity_arbiter.py
    ├── stm32_communicator.py
    ├── manual_bridge.py
    ├── ...
    ├── models/                       # ONNX, Caffe models
    │   ├── mb2_gap.onnx
    │   ├── MobileNetSSD_deploy.prototxt
    │   └── MobileNetSSD_deploy.caffemodel
    ├── sounds/
    └── data/
```

### 1.2 Thành Phần Trong person_detector.py Hiện Tại

| Dòng | Thành phần | Mô tả |
|------|-----------|-------|
| 1-36 | **Imports** | ROS2, OpenCV, ONNX, scipy |
| 37-44 | **Paths** | Đường dẫn models |
| 46-113 | **Helper functions** | `iou()`, `clamp()`, `create_tracker()`, overlay helpers |
| 114-207 | **ReID Features** | `enhanced_body_feature()`, HSV histogram, depth feature |
| 208-229 | **Detector** | MobileNet-SSD detection |
| 232-382 | **PersonDetector.__init__** | Khởi tạo node, params, publishers |
| 384-416 | **Depth processing** | `get_median_depth_at_box()`, `is_target_occluded()` |
| 417-449 | **Auto-enroll** | Thu thập mẫu target |
| 451-492 | **Control** | `compute_cmd()` - điều khiển robot |
| **498-521** | **CSRT Tracker** | `init_tracker()`, `update_tracker()` ← **SẼ XÓA** |
| 523-550 | **Sound** | Lost sound loop |
| 552-579 | **Matching** | `find_best_match_by_reid()`, `find_best_match_by_iou()` |
| 581-621 | **Adaptive update** | Cập nhật model ReID |
| 623-688 | **Debug** | `publish_debug()` |
| 690-846 | **on_image callback** | State machine chính |
| 849-857 | **main()** | Entry point |

### 1.3 State Machine Hiện Tại

```
AUTO-ENROLL → SEARCHING → LOCKED ⇄ LOST
                 ↑______________|
```

| State | Xử lý tracking |
|-------|----------------|
| SEARCHING | Duyệt tất cả detections, so ReID |
| LOCKED | IoU match + **CSRT fallback** |
| LOST | **CSRT predict** + grace period |

---

## 2. Vấn Đề Với CSRT Tracker

### 2.1 Code CSRT Hiện Tại (Dòng 498-521)

```python
def create_tracker():
    for cand in ["legacy.TrackerCSRT_create","TrackerCSRT_create",
                 "legacy.TrackerKCF_create","TrackerKCF_create",
                 "legacy.TrackerMOSSE_create","TrackerMOSSE_create"]:
        c=_get_ctor(cand)
        if callable(c):
            try: return c()
            except Exception: continue
    return None

def init_tracker(self, frame, box):
    self.tracker = create_tracker()
    if self.tracker:
        x1,y1,x2,y2 = box
        self.tracker.init(frame, (x1, y1, x2-x1, y2-y1))

def update_tracker(self, frame):
    if self.tracker:
        ok, box = self.tracker.update(frame)
        if ok:
            x, y, w, h = map(int, box)
            return (x, y, x+w, y+h)
```

### 2.2 Vấn Đề

| Vấn đề | Mô tả | Hậu quả |
|--------|-------|---------|
| **Drift** | CSRT dùng correlation filter, dễ bám vào background | Target bị mất khi đứng yên |
| **No motion model** | Không predict được vị trí tiếp theo | Mất target khi di chuyển nhanh |
| **Fixed template** | Không adapt với thay đổi appearance | Mất khi người xoay người |
| **No velocity** | Không biết target đang đi hướng nào | Không predict được |
| **Single template** | Chỉ dùng 1 mẫu ban đầu | Không robust |

---

## 3. DeepSORT Giải Quyết Như Thế Nào

### 3.1 Kiến Trúc DeepSORT

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepSORT Tracker                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Kalman      │    │ Cost Matrix  │    │ Hungarian     │  │
│  │ Filter      │───>│ Motion +     │───>│ Matching      │  │
│  │ (8-dim)     │    │ Appearance   │    │ (Optimal)     │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                                       │          │
│         v                                       v          │
│  ┌─────────────┐                        ┌───────────────┐  │
│  │ Track       │<───────────────────────│ Track         │  │
│  │ Manager     │                        │ Update/Create │  │
│  └─────────────┘                        └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 So Sánh CSRT vs DeepSORT

| Tiêu chí | CSRT | DeepSORT |
|----------|------|----------|
| Motion model | ❌ Không có | ✅ Kalman Filter 8-dim |
| Velocity tracking | ❌ | ✅ vx, vy, va, vh |
| Appearance matching | ❌ Correlation | ✅ ReID + Cosine similarity |
| Multi-object | ❌ Single | ✅ Multi (nhưng ta dùng single) |
| Re-identification | ❌ | ✅ Feature history |
| Prediction khi mất | ❌ Dựa vào template | ✅ Kalman predict |
| Occlusion handling | ❌ Yếu | ✅ Mạnh |

---

## 4. Cấu Trúc File Sau Khi Implement

### 4.1 Cấu Trúc Mới

```
mecanum_control/
├── setup.py                          # [MODIFY] Thêm sub-package
├── package.xml
├── launch/
│   └── mecanum.launch.py             # [KHÔNG ĐỔI]
│
└── mecanum_control/
    ├── __init__.py
    ├── person_detector.py            # [MODIFY] Dùng DeepSORT
    │
    ├── tracking/                     # [NEW] Sub-package DeepSORT
    │   ├── __init__.py               # [NEW] Export classes
    │   ├── kalman_filter.py          # [NEW] Kalman Filter 8-dim
    │   ├── track.py                  # [NEW] Track class
    │   ├── tracker.py                # [NEW] DeepSORTTracker
    │   └── nn_matching.py            # [NEW] Cost matrix utils
    │
    ├── models/                       # [KHÔNG ĐỔI]
    ├── sounds/                       # [KHÔNG ĐỔI]
    └── data/                         # [KHÔNG ĐỔI]
```

### 4.2 Tóm Tắt Thay Đổi

| File | Action | Số dòng (ước tính) |
|------|--------|-------------------|
| `tracking/__init__.py` | NEW | ~10 |
| `tracking/kalman_filter.py` | NEW | ~120 |
| `tracking/track.py` | NEW | ~100 |
| `tracking/tracker.py` | NEW | ~200 |
| `tracking/nn_matching.py` | NEW | ~80 |
| `person_detector.py` | MODIFY | -50, +30 |
| `setup.py` | MODIFY | +1 |

**Tổng**: ~510 dòng code mới, thay thế ~50 dòng CSRT

---

## 5. Chi Tiết Từng File Mới

### 5.1 `tracking/__init__.py`

```python
"""
DeepSORT Tracking Sub-package
Optimized for single-target person tracking on CPU
"""

from .tracker import DeepSORTTracker
from .track import Track, TrackState
from .kalman_filter import KalmanFilter

__all__ = ['DeepSORTTracker', 'Track', 'TrackState', 'KalmanFilter']
```

---

### 5.2 `tracking/kalman_filter.py`

**Mục đích**: Kalman Filter 8-dimensional cho tracking

**State Vector** (8-dim):
```
[x, y, a, h, vx, vy, va, vh]
 │  │  │  │   │   │   │   │
 │  │  │  │   │   │   │   └── velocity of height
 │  │  │  │   │   │   └────── velocity of aspect ratio
 │  │  │  │   │   └────────── velocity of y
 │  │  │  │   └──────────────  velocity of x
 │  │  │  └────────────────── height
 │  │  └─────────────────────  aspect ratio (w/h)
 │  └────────────────────────  center y
 └───────────────────────────  center x
```

**Measurement** (4-dim): `[x, y, a, h]`

**Methods**:

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `initiate(measurement)` | `[x,y,a,h]` | `mean, cov` | Khởi tạo track mới |
| `predict(mean, cov)` | previous state | predicted state | Predict vị trí tiếp |
| `update(mean, cov, measurement)` | state + detection | updated state | Update với detection |
| `gating_distance(mean, cov, measurements)` | state + nhiều detections | distances | Mahalanobis distance |

**Code Structure**:
```python
class KalmanFilter:
    def __init__(self):
        # Motion matrix F (8x8)
        # Measurement matrix H (4x8)
        # Process noise Q
        # Measurement noise R
        
    def initiate(self, measurement):
        # mean = [x, y, a, h, 0, 0, 0, 0]
        # covariance = initial uncertainty
        
    def predict(self, mean, covariance):
        # mean = F @ mean
        # covariance = F @ cov @ F.T + Q
        
    def update(self, mean, covariance, measurement):
        # Kalman gain K
        # mean = mean + K @ (measurement - H @ mean)
        # covariance = (I - K @ H) @ covariance
        
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        # Mahalanobis distance for gating
```

---

### 5.3 `tracking/track.py`

**Mục đích**: Quản lý một track đơn lẻ

**TrackState Enum**:
```python
class TrackState:
    Tentative = 1   # Chưa xác nhận (mới tạo)
    Confirmed = 2   # Đã xác nhận (đủ hits)
    Deleted = 3     # Đã xóa (quá lâu không update)
```

**Track Class**:

| Attribute | Type | Mô tả |
|-----------|------|-------|
| `track_id` | int | ID duy nhất |
| `mean` | ndarray | Kalman state [8] |
| `covariance` | ndarray | Kalman covariance [8x8] |
| `hits` | int | Số lần match liên tiếp |
| `age` | int | Số frames tồn tại |
| `time_since_update` | int | Frames kể từ lần update cuối |
| `state` | TrackState | Trạng thái hiện tại |
| `features` | list | Lịch sử feature (max 30) |

**Methods**:

| Method | Mô tả |
|--------|-------|
| `predict()` | Dùng Kalman filter predict vị trí |
| `update(detection, feature)` | Update state với detection mới |
| `mark_missed()` | Gọi khi không match được |
| `is_tentative()` | Kiểm tra state |
| `is_confirmed()` | Kiểm tra state |
| `is_deleted()` | Kiểm tra state |
| `to_tlbr()` | Convert state → `[x1,y1,x2,y2]` |

**Lifecycle**:
```
                    hits >= n_init
    [NEW] ──────────────────────────> [Confirmed]
      │                                    │
      │ time_since_update > max_age        │ time_since_update > max_age
      v                                    v
  [Deleted] <─────────────────────── [Deleted]
```

---

### 5.4 `tracking/tracker.py`

**Mục đích**: Main DeepSORT tracker class

**Tham số khởi tạo**:

| Param | Default | Mô tả |
|-------|---------|-------|
| `max_age` | 30 | Max frames không update trước khi xóa |
| `n_init` | 3 | Số hits để confirm track |
| `max_cosine_distance` | 0.4 | Ngưỡng cosine distance |
| `nn_budget` | 30 | Max số features lưu mỗi track |

**Methods**:

| Method | Mô tả |
|--------|-------|
| `predict()` | Kalman predict tất cả tracks |
| `update(detections, features)` | Main update loop |
| `_match(detections, features)` | Cascade matching |
| `_initiate_track(detection, feature)` | Tạo track mới |

**Update Algorithm** (mỗi frame):
```
1. PREDICT: Kalman predict cho tất cả tracks
   
2. MATCH CONFIRMED TRACKS:
   - Tính cost matrix = λ*motion + (1-λ)*appearance
   - Gating: loại cặp có distance > threshold
   - Hungarian matching
   
3. MATCH TENTATIVE TRACKS:
   - Chỉ dùng IoU (không dùng appearance)
   - Hungarian matching
   
4. UPDATE MATCHED TRACKS:
   - Kalman update với detection
   - Thêm feature vào history
   
5. HANDLE UNMATCHED:
   - Unmatched tracks: mark_missed()
   - Unmatched detections: _initiate_track()
   
6. CLEANUP:
   - Xóa tracks đã deleted
```

---

### 5.5 `tracking/nn_matching.py`

**Mục đích**: Nearest Neighbor matching utilities

**Functions**:

| Function | Mô tả |
|----------|-------|
| `_cosine_distance(a, b)` | Cosine distance giữa 2 feature vectors |
| `_nn_cosine_distance(x, y)` | Min cosine distance từ x đến tất cả y |
| `iou(bbox, candidates)` | IoU giữa 1 box và nhiều candidates |
| `iou_cost(tracks, detections)` | IoU cost matrix |
| `gate_cost_matrix(kf, cost_matrix, tracks, detections)` | Apply Mahalanobis gating |

**Cost Matrix**:
```python
# Combined cost
cost = lambda_weight * motion_cost + (1 - lambda_weight) * appearance_cost

# Gating (loại các cặp không hợp lý)
INFINITY = 1e5
cost[motion_distance > chi2_threshold] = INFINITY
cost[appearance_distance > max_cosine_distance] = INFINITY
```

---

## 6. Thay Đổi Trong person_detector.py

### 6.1 Thêm Import

```diff
+ from mecanum_control.tracking import DeepSORTTracker
```

### 6.2 Xóa CSRT Functions (Dòng 68-83, 498-521)

```diff
- def _get_ctor(path):
-     cur = cv2
-     for name in path.split('.'):
-         if not hasattr(cur, name): return None
-         cur = getattr(cur, name)
-     return cur
-
- def create_tracker():
-     for cand in ["legacy.TrackerCSRT_create","TrackerCSRT_create",...]:
-         c=_get_ctor(cand)
-         ...
-     return None

- def init_tracker(self, frame, box):
-     self.tracker = create_tracker()
-     if self.tracker:
-         x1,y1,x2,y2 = box
-         self.tracker.init(frame, (x1, y1, x2-x1, y2-y1))
-
- def update_tracker(self, frame):
-     if self.tracker:
-         ok, box = self.tracker.update(frame)
-         if ok:
-             ...
```

### 6.3 Thêm Biến __init__ (Khoảng dòng 340)

```diff
  # --- STATE MACHINE VARIABLES ---
  self.state = 'AUTO-ENROLL'
  self.target_box = None
  self.target_feature = None
  self.last_known_depth = None
- self.tracker = None
+ 
+ # DeepSORT Tracker
+ self.deepsort = DeepSORTTracker(
+     max_age=30,
+     n_init=3,
+     max_cosine_distance=0.4
+ )
+ self.current_track_id = None  # ID của target track
  self.lost_start_time = None
```

### 6.4 Thay Đổi on_image Callback

**Hiện tại (LOCKED state, dòng 751-809)**:
```python
elif self.state == 'LOCKED':
    # ... occlusion check ...
    
    current_box, current_score = self.find_best_match_by_iou(...)
    
    if current_box and current_score > reject_thr:
        self.target_box = current_box
        self.init_tracker(frame, self.target_box)  # ← CSRT
    else:
        tracker_box = self.update_tracker(frame)   # ← CSRT fallback
        if tracker_box:
            # verify and use
        else:
            self.state = 'LOST'
```

**Sau khi thay đổi**:
```python
elif self.state == 'LOCKED':
    # ... occlusion check ...
    
    # DeepSORT update
    features = [enhanced_body_feature(frame, box, depth_frame, ...) 
                for box in pboxes]
    tracks = self.deepsort.update(pboxes, features)
    
    # Tìm track của target
    target_track = None
    for track in tracks:
        if track.is_confirmed() and track.track_id == self.current_track_id:
            target_track = track
            break
    
    if target_track is not None:
        self.target_box = target_track.to_tlbr()
        self.last_known_depth = self.get_median_depth_at_box(...)
    else:
        # Target track lost, try to find by ReID
        best_track = self._find_best_track_by_reid(tracks)
        if best_track:
            self.current_track_id = best_track.track_id
            self.target_box = best_track.to_tlbr()
        else:
            self.state = 'LOST'
            self.lost_start_time = time.time()
```

**Thay đổi LOST state (dòng 811-831)**:
```python
elif self.state == 'LOST':
    # DeepSORT predict (Kalman)
    self.deepsort.predict()
    
    # Check nếu track vẫn còn tồn tại
    target_track = None
    for track in self.deepsort.tracks:
        if track.track_id == self.current_track_id and not track.is_deleted():
            target_track = track
            break
    
    if target_track:
        # Target vẫn được predict bởi Kalman
        self.target_box = target_track.to_tlbr()
        
        # Try to re-acquire với detection mới
        if target_track.is_confirmed():
            self.state = 'LOCKED'
            self.stop_lost_sound_loop()
    else:
        # Track đã bị xóa
        if time.time() - self.lost_start_time > grace_period:
            self.state = 'SEARCHING'
            self.current_track_id = None
            self.start_lost_sound_loop()
```

---

## 7. Thay Đổi Trong setup.py

```diff
  setup(
      name=package_name,
      version='0.0.0',
      packages=[
          package_name,
+         f'{package_name}.tracking',
      ],
      package_data={
          package_name: [
              'models/*',
              'sounds/*',
              'data/*',
          ],
      },
      ...
  )
```

---

## 8. Tham Số DeepSORT

### 8.1 Tham Số Có Thể Tune

| Param | Default | Range | Mô tả |
|-------|---------|-------|-------|
| `max_age` | 30 | 15-60 | Frames giữ track khi mất |
| `n_init` | 3 | 2-5 | Hits để confirm |
| `max_cosine_distance` | 0.4 | 0.2-0.6 | Ngưỡng appearance |
| `lambda_weight` | 0.3 | 0.0-1.0 | Motion vs Appearance weight |
| `nn_budget` | 30 | 10-100 | Max features lưu |

### 8.2 Tham Số Kalman Filter (Cố Định)

| Param | Value | Mô tả |
|-------|-------|-------|
| `chi2_threshold` | 9.4877 | Chi-square 95% (4 DOF) |
| `std_weight_position` | 1/20 | Uncertainty của position |
| `std_weight_velocity` | 1/160 | Uncertainty của velocity |

---

## 9. So Sánh Trước/Sau

### 9.1 Logic State Machine

| State | Trước (CSRT) | Sau (DeepSORT) |
|-------|--------------|----------------|
| **SEARCHING** | Duyệt boxes, so ReID | Duyệt tracks, so ReID |
| **LOCKED** | IoU match → CSRT fallback | Track ID match → Kalman predict |
| **LOST** | CSRT predict only | Kalman predict + feature match |

### 9.2 Khi Target Bị Che 2 Giây

| Bước | CSRT | DeepSORT |
|------|------|----------|
| Frame 1-30 | CSRT predict (có thể drift) | Kalman predict (smooth) |
| Frame 31+ | Mất hoàn toàn | Kalman vẫn predict |
| Xuất hiện lại | Phải SEARCHING lại | ReID match ngay |

### 9.3 Performance Estimate

| Metric | CSRT | DeepSORT |
|--------|------|----------|
| CPU/frame (2-3 người) | ~15ms | ~20ms (+5ms Kalman) |
| Memory | ~5MB | ~10MB (feature history) |
| Re-identification | ❌ | ✅ |
| Occlusion handling | ⭐ | ⭐⭐⭐⭐ |

---

## 10. Cách Chạy (Không Đổi)

### 10.1 Build

```bash
cd ~/backup_16_12_2025/ros2_ws
colcon build --packages-select mecanum_control
source install/setup.bash
```

### 10.2 Launch (GIỐNG HỆT TRƯỚC)

```bash
ros2 launch mecanum_control mecanum.launch.py
```

### 10.3 Kiểm Tra

```bash
# Xem state
ros2 topic echo /person_detector/follow_state

# Xem debug image
ros2 run image_view image_view --ros-args -r image:=/person_detector/debug_image
```

---

## 11. Checklist Trước Khi Implement

- [ ] Bạn đã đọc và hiểu kế hoạch này
- [ ] Đồng ý với cấu trúc file mới (tracking/ sub-package)
- [ ] Đồng ý thay thế hoàn toàn CSRT
- [ ] Hiểu rằng launch file KHÔNG thay đổi
- [ ] Sẵn sàng test sau khi implement

---

> **Tiếp theo**: Sau khi bạn confirm OK, tôi sẽ bắt đầu implement từng file theo thứ tự:
> 1. `tracking/kalman_filter.py`
> 2. `tracking/track.py`
> 3. `tracking/nn_matching.py`
> 4. `tracking/tracker.py`
> 5. `tracking/__init__.py`
> 6. Sửa `setup.py`
> 7. Sửa `person_detector.py`

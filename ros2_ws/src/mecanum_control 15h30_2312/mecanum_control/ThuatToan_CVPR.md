# Single-Target Person Following với Enhanced DeepSORT
## Phân Tích Thuật Toán Chi Tiết Theo Chuẩn CVPR

---

## 1. Tổng Quan Hệ Thống (System Overview)

### 1.1 Pipeline Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SINGLE-TARGET FOLLOWING SYSTEM                    │
├─────────────────────────────────────────────────────────────────────────┤
│  RGB Frame                                                              │
│      ↓                                                                  │
│  ┌──────────────────┐                                                   │
│  │ 1. DETECTION     │  MobileNet-SSD → Bounding Boxes                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 2. FEATURE       │  MobileNetV2 + HSV + Depth → Feature Vector       │
│  │    EXTRACTION    │                                                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 3. DATA          │  Hungarian Algorithm + Cost Matrix                │
│  │    ASSOCIATION   │                                                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 4. MOTION        │  Kalman Filter (8D State Space)                   │
│  │    PREDICTION    │                                                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 5. TRACK & TARGET│  State Machine + ReID Selection                   │
│  │    MANAGEMENT    │                                                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 6. ONLINE        │  Auto-Enroll + Anchor-Based Update                │
│  │    ADAPTATION    │                                                   │
│  └────────┬─────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ 7. ROBOT CONTROL │  P-Control + Depth EMA                            │
│  └──────────────────┘                                                   │
│           ↓                                                             │
│      Twist Command (vx, wz)                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Ký Hiệu Toán Học

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\mathbf{x}_t$ | State vector tại thời điểm $t$ |
| $\mathbf{z}_t$ | Measurement (observation) tại thời điểm $t$ |
| $\mathbf{f}_i$ | Feature vector của detection $i$ |
| $\mathbf{f}^*$ | Target feature (anchor) |
| $\mathcal{T}$ | Tập hợp các tracks |
| $\mathcal{D}$ | Tập hợp các detections |

---

## 2. Module 1: Detection (MobileNet-SSD)

### 2.1 Kiến Trúc

```
Input Image (H×W×3)
       ↓
   Resize (300×300)
       ↓
   Blob Creation (mean subtraction)
       ↓
┌──────────────────────────────┐
│   MobileNet-SSD Backbone     │
│   - Depthwise Separable Conv │
│   - SSD Detection Head       │
└──────────────────────────────┘
       ↓
   NMS + Confidence Thresholding
       ↓
   Bounding Boxes [(x1,y1,x2,y2), ...]
```

### 2.2 Công Thức Tiền Xử Lý

$$\mathbf{I}_{blob} = \frac{\mathbf{I}_{resized} - 127.5}{127.5}$$

Trong đó:
- $\mathbf{I}_{resized}$: Ảnh đã resize về 300×300
- Scale factor: 0.007843 (≈ 1/127.5)

### 2.3 Confidence Filtering

Chỉ giữ detections với:
$$\text{conf}(d_i) > \tau_{conf} \quad \text{và} \quad \text{class}(d_i) = 15 \text{ (person)}$$

Với $\tau_{conf} = 0.35$ (ngưỡng confidence).

### 2.4 Code Reference

```python
# File: person_detector.py, Line 216-231
def _ssd_detect(net, frame, conf_thresh=0.4):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300,300)), 
        0.007843,      # scale
        (300,300),     # size
        127.5          # mean
    )
    net.setInput(blob)
    det = net.forward()
    # Filter: class==15 (person) AND conf > threshold
```

---

## 3. Module 2: Feature Extraction (ReID)

### 3.1 Kiến Trúc Multi-Modal Feature

Hệ thống sử dụng **3 loại đặc trưng** kết hợp:

```
┌─────────────────────────────────────────────────────────────┐
│                   ENHANCED BODY FEATURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │  MobileNetV2│     │     HSV     │     │    Depth    │   │
│  │  Embedding  │     │  Histogram  │     │   Feature   │   │
│  │   (1280-D)  │     │   (48-D)    │     │   (256-D)   │   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘   │
│         │                   │                   │          │
│         │ × (1-w_c)         │ × w_c             │ × 0.1    │
│         └─────────┬─────────┴─────────┬─────────┘          │
│                   ↓                   ↓                    │
│              Concatenate + L2 Normalize                    │
│                        ↓                                   │
│               Final Feature (1584-D)                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Shape Feature (MobileNetV2)

**Preprocessing Keras-style:**
$$\mathbf{x}_{preprocessed} = \frac{\mathbf{x}_{RGB}}{127.5} - 1.0$$

**Embedding extraction:**
$$\mathbf{e}_{shape} = \text{MobileNetV2-GAP}(\mathbf{x}_{preprocessed}) \in \mathbb{R}^{1280}$$

**L2 Normalization:**
$$\hat{\mathbf{e}}_{shape} = \frac{\mathbf{e}_{shape}}{\|\mathbf{e}_{shape}\|_2 + \epsilon}$$

### 3.3 Color Feature (HSV Histogram)

**HSV Histogram với brightness normalization:**

1. **Normalize brightness:**
$$V_{normalized} = \min\left(\frac{V \times 128}{\bar{V}}, 255\right)$$

2. **Compute histograms:**
$$\mathbf{h}_H = \text{hist}(H, \text{bins}=16, \text{range}=[0,180])$$
$$\mathbf{h}_S = \text{hist}(S, \text{bins}=16, \text{range}=[0,256])$$
$$\mathbf{h}_V = \text{hist}(V, \text{bins}=16, \text{range}=[0,256]) \times w_V$$

3. **Concatenate:**
$$\mathbf{e}_{color} = [\mathbf{h}_H; \mathbf{h}_S; \mathbf{h}_V] \in \mathbb{R}^{48}$$

Với $w_V = 0.6$ (giảm trọng số kênh V để chống nhiễu ánh sáng).

### 3.4 Depth Feature

**Trích xuất depth map:**
$$\mathbf{D}_{roi} = \text{resize}(\mathbf{D}[y_1:y_2, x_1:x_2], (16, 16))$$

**Normalization (gần → 1, xa → 0):**
$$\mathbf{e}_{depth} = \text{clip}\left(\frac{5000 - \mathbf{D}_{roi}}{4500}, 0, 1\right)$$

### 3.5 Feature Fusion

**Công thức kết hợp:**
$$\mathbf{f} = \frac{[\hat{\mathbf{e}}_{shape} \times (1-w_c); \hat{\mathbf{e}}_{color} \times w_c; \hat{\mathbf{e}}_{depth} \times 0.1]}{\|[\cdot]\|_2}$$

Với $w_c = 0.22$ (color weight, giảm khi ánh sáng yếu/mạnh).

### 3.6 Dynamic Color Weight Adjustment

```python
if vmean < 90 or vmean > 200:  # Low-light or backlit
    w_c = min(0.10, base_w_c × 0.6)
else:
    w_c = base_w_c  # 0.22
```

---

## 4. Module 3: Data Association

### 4.1 Two-Stage Matching (Cascade Matching)

```
┌─────────────────────────────────────────────────────────────┐
│                    MATCHING CASCADE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: CONFIRMED TRACKS                                  │
│  ────────────────────────                                   │
│  Cost = λ × IoU_cost + (1-λ) × Appearance_cost              │
│  + Kalman Gating + Appearance Threshold                     │
│  → Hungarian Algorithm                                      │
│                                                             │
│  Stage 2: TENTATIVE TRACKS + UNMATCHED                      │
│  ─────────────────────────────────────                      │
│  Cost = IoU_cost only                                       │
│  → Hungarian Algorithm                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Cost Matrix Computation

**Appearance Cost (Cosine Distance):**
$$C_{app}(i,j) = 1 - \frac{\mathbf{f}_j^T \cdot \bar{\mathbf{f}}_i}{\|\mathbf{f}_j\|_2 \cdot \|\bar{\mathbf{f}}_i\|_2}$$

Trong đó:
- $\mathbf{f}_j$: Feature của detection $j$
- $\bar{\mathbf{f}}_i$: Mean feature của track $i$ (từ history)

**IoU Cost:**
$$C_{IoU}(i,j) = 1 - \text{IoU}(\mathbf{b}_i, \mathbf{d}_j)$$

$$\text{IoU}(\mathbf{b}, \mathbf{d}) = \frac{|\mathbf{b} \cap \mathbf{d}|}{|\mathbf{b} \cup \mathbf{d}|}$$

**Combined Cost:**
$$C(i,j) = \lambda \cdot C_{IoU}(i,j) + (1-\lambda) \cdot C_{app}(i,j)$$

Với $\lambda = 0.3$ (motion weight).

### 4.3 Kalman Gating

Loại bỏ các cặp (track, detection) có khoảng cách Mahalanobis quá lớn:

$$d^2(\mathbf{x}_i, \mathbf{z}_j) = (\mathbf{z}_j - H\mathbf{x}_i)^T S_i^{-1} (\mathbf{z}_j - H\mathbf{x}_i)$$

$$C(i,j) = \infty \quad \text{nếu} \quad d^2 > \chi^2_{0.95,4} = 9.4877$$

### 4.4 Hungarian Algorithm

Giải bài toán Linear Assignment:
$$\min_{\pi} \sum_{i} C(i, \pi(i))$$

**Code:**
```python
row_indices, col_indices = linear_sum_assignment(cost_matrix)
```

---

## 5. Module 4: Motion Prediction (Kalman Filter)

### 5.1 State Space Model

**State vector (8D):**
$$\mathbf{x} = [x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]^T$$

Trong đó:
- $(x, y)$: Tâm bounding box
- $a = w/h$: Aspect ratio
- $h$: Chiều cao
- $(\dot{x}, \dot{y}, \dot{a}, \dot{h})$: Vận tốc tương ứng

**Measurement vector (4D):**
$$\mathbf{z} = [x, y, a, h]^T$$

### 5.2 Motion Model (Constant Velocity)

**Transition Matrix:**
$$F = \begin{bmatrix} I_4 & \Delta t \cdot I_4 \\ 0 & I_4 \end{bmatrix}$$

**Observation Matrix:**
$$H = \begin{bmatrix} I_4 & 0 \end{bmatrix}$$

### 5.3 Predict Step

$$\hat{\mathbf{x}}_{t|t-1} = F \mathbf{x}_{t-1|t-1}$$
$$\hat{P}_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

Với process noise $Q$:
$$Q = \text{diag}(\sigma_{pos}^2, \sigma_{pos}^2, \sigma_a^2, \sigma_{pos}^2, \sigma_{vel}^2, \sigma_{vel}^2, \sigma_{\dot{a}}^2, \sigma_{vel}^2)$$

### 5.4 Update Step

**Kalman Gain:**
$$K = \hat{P}_{t|t-1} H^T (H \hat{P}_{t|t-1} H^T + R)^{-1}$$

**State Update:**
$$\mathbf{x}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K(\mathbf{z}_t - H\hat{\mathbf{x}}_{t|t-1})$$

**Covariance Update:**
$$P_{t|t} = (I - KH) \hat{P}_{t|t-1}$$

---

## 6. Module 5: Track & Target Management

### 6.1 Track Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│                    TRACK STATE MACHINE                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   [New Detection]                                        │
│         ↓                                                │
│   ┌─────────────┐                                        │
│   │  TENTATIVE  │ ←── hits < n_init                      │
│   └──────┬──────┘                                        │
│          │ hits >= n_init                                │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │  CONFIRMED  │ ←── Active tracking                    │
│   └──────┬──────┘                                        │
│          │ time_since_update > max_age                   │
│          ↓                                               │
│   ┌─────────────┐                                        │
│   │   DELETED   │ ←── Remove from tracker                │
│   └─────────────┘                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Parameters:**
- `n_init = 3`: Số frame liên tiếp để confirm
- `max_age = 30`: Số frame tối đa không match trước khi xóa

### 6.2 Feature History Management

```python
# Mỗi track lưu tối đa 30 features gần nhất
if len(self.features) > 30:
    self.features = self.features[-30:]

# Mean feature cho matching
def get_feature(self):
    return np.mean(self.features, axis=0)
```

### 6.3 Target Selection State Machine (Tích hợp Audio Feedback)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            SINGLE-TARGET STATE MACHINE với AUDIO FEEDBACK                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                      AUTO-ENROLL                             │            │
│  │  🔊 enroll_viet.wav (2x) - "Mời bạn đứng trước camera..."    │            │
│  │  📷 Thu thập samples → Tính centroid → Lưu anchor            │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │ enrollment done                               │
│                             │ (timeout OR samples >= target)                │
│                             ↓                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                       SEARCHING                              │            │
│  │  🔊 run_viet.wav (2x) - "Bắt đầu theo dõi..."               │            │
│  │  🔍 Tìm track có similarity > τ_accept với target_feature    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │ best_track found                              │
│                             │ similarity > τ_accept = 0.75                  │
│                             ↓                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                        LOCKED                                │            │
│  │  🔇 Stop lost sound (nếu đang phát)                          │            │
│  │  🎯 Theo dõi target_track, cập nhật box, điều khiển robot    │            │
│  │  📊 Adaptive model update nếu điều kiện thỏa mãn             │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                           │
│         │ similarity < τ_reject = 0.6           │ occlusion detected        │
│         │ OR track deleted                      │ OR track.time_since > 0   │
│         ↓                                       ↓                           │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                         LOST                                 │            │
│  │  ⏳ Grace period = 2.0s (chờ re-acquire)                     │            │
│  │  🔮 Kalman predict vị trí target (không có detection)        │            │
│  │  🔍 Tìm lại bằng ReID trong confirmed tracks                 │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                           │
│         │ re-acquire thành công                 │ grace_period expired      │
│         │ similarity > τ_accept                 │                           │
│         ↓                                       ↓                           │
│  ┌──────────────┐                    ┌─────────────────────────┐            │
│  │    LOCKED    │                    │       SEARCHING         │            │
│  │ 🔇 Stop sound│                    │ 🔊 lost_target_viet.wav │            │
│  └──────────────┘                    │    (LOOP liên tục)      │            │
│                                      └─────────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Chi Tiết Từng State với Audio

#### **STATE 1: AUTO-ENROLL**

| Thuộc tính | Giá trị |
|------------|---------|
| **Mục đích** | Thu thập mẫu đặc trưng của target |
| **Audio** | 🔊 `enroll_viet.wav` × 2 lần (non-blocking) |
| **Điều kiện vào** | Khởi động hệ thống |
| **Điều kiện ra** | `timeout >= 30s` OR `samples >= 100` |
| **Robot action** | Đứng yên, không di chuyển |

```python
# Pseudo-code
if state == 'AUTO-ENROLL':
    # 1. Phát audio hướng dẫn (1 lần duy nhất)
    if not enroll_audio_played:
        play_async("enroll_viet.wav", repeat=2)
        enroll_audio_played = True
    
    # 2. Thu thập samples
    if person_detected:
        feature = enhanced_body_feature(frame, box, depth)
        body_samples.append(feature)
        body_centroid = 0.9 * body_centroid + 0.1 * feature  # EMA
    
    # 3. Kiểm tra điều kiện hoàn thành
    if time_elapsed >= timeout OR len(samples) >= target:
        target_feature = body_centroid.copy()
        original_target_feature = body_centroid.copy()  # ANCHOR
        state = 'SEARCHING'
```

---

#### **STATE 2: SEARCHING**

| Thuộc tính | Giá trị |
|------------|---------|
| **Mục đích** | Tìm target trong các confirmed tracks |
| **Audio (từ ENROLL)** | 🔊 `run_viet.wav` × 2 lần |
| **Audio (từ LOST)** | 🔊 `lost_target_viet.wav` LOOP |
| **Điều kiện vào** | Enrollment xong HOẶC Grace period hết |
| **Điều kiện ra** | Tìm thấy track với `similarity > 0.75` |
| **Robot action** | Đứng yên, quét tìm |

```python
# Pseudo-code
if state == 'SEARCHING':
    # 1. Phát audio tương ứng
    if just_finished_enrollment:
        if not run_audio_played:
            play_async("run_viet.wav", repeat=2)
            run_audio_played = True
    elif from_lost_state:
        start_lost_sound_loop()  # Loop liên tục
    
    # 2. Tìm best track bằng ReID
    best_track = None
    best_score = -1
    for track in confirmed_tracks:
        score = cosine_similarity(track.feature, target_feature)
        if score > best_score:
            best_score = score
            best_track = track
    
    # 3. Chuyển state nếu tìm thấy
    if best_score > τ_accept:  # 0.75
        state = 'LOCKED'
        current_track_id = best_track.track_id
        stop_lost_sound_loop()  # Dừng phát lost sound
```

---

#### **STATE 3: LOCKED**

| Thuộc tính | Giá trị |
|------------|---------|
| **Mục đích** | Theo dõi target, điều khiển robot |
| **Audio** | 🔇 Dừng lost sound (nếu đang phát) |
| **Điều kiện vào** | Tìm thấy track với `similarity > 0.75` |
| **Điều kiện ra** | `similarity < 0.6` OR track deleted OR occlusion |
| **Robot action** | Điều khiển heading + distance |

```python
# Pseudo-code
if state == 'LOCKED':
    # 1. Dừng lost sound khi lock được target
    stop_lost_sound_loop()
    
    # 2. Kiểm tra occlusion bằng depth
    if is_target_occluded(target_box, depth, last_known_depth):
        state = 'LOST'
        lost_start_time = now()
        return
    
    # 3. Lấy target track
    target_track = deepsort.get_track_by_id(current_track_id)
    
    if target_track is not None:
        target_box = target_track.to_tlbr()
        
        # 4. Tính similarity
        similarity = cosine(track.feature, target_feature)
        current_similarity = similarity  # Hiển thị lên UI
        
        # 5. Adaptive model update
        if similarity > τ_reject AND similarity < 0.99:
            adaptive_model_update(target_box, frame, depth)
        
        # 6. Kiểm tra mất target
        if similarity < τ_reject:  # 0.6
            state = 'LOST'
            lost_start_time = now()
    else:
        # Track không còn, chuyển LOST
        state = 'LOST'
        lost_start_time = now()
    
    # 7. Điều khiển robot
    twist = compute_cmd(frame_w, frame_h, target_box)
    publish(twist)
```

---

#### **STATE 4: LOST**

| Thuộc tính | Giá trị |
|------------|---------|
| **Mục đích** | Chờ re-acquire trong grace period |
| **Audio** | Không phát ngay (chờ hết grace period) |
| **Điều kiện vào** | Similarity thấp OR track deleted OR occlusion |
| **Điều kiện ra** | Re-acquire thành công OR grace_period hết |
| **Robot action** | Đứng yên, sử dụng Kalman predict |

```python
# Pseudo-code
if state == 'LOST':
    # 1. Kalman vẫn predict vị trí (dù không có detection)
    target_track = deepsort.get_track_by_id(current_track_id)
    
    if target_track is not None:
        target_box = target_track.to_tlbr()  # Kalman predicted box
        
        # 2. Kiểm tra re-acquire
        if target_track.time_since_update == 0:  # Matched với detection
            similarity = cosine(track.feature, target_feature)
            if similarity > τ_accept:  # 0.75
                state = 'LOCKED'
                stop_lost_sound_loop()
                return
    else:
        # 3. Thử tìm bằng ReID
        best_track = find_best_track_by_reid(confirmed_tracks)
        if best_track is not None:
            state = 'LOCKED'
            current_track_id = best_track.track_id
            stop_lost_sound_loop()
            return
    
    # 4. Kiểm tra grace period
    if now() - lost_start_time > grace_period:  # 2.0s
        state = 'SEARCHING'
        target_box = None
        current_track_id = None
        start_lost_sound_loop()  # 🔊 Bắt đầu phát lost sound LOOP
```

---

### 6.5 Audio Event Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIO TRIGGERS TIMELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  t=0    t=30s              t=32s        t=45s         t=47s                 │
│   │       │                  │            │             │                   │
│   ▼       ▼                  ▼            ▼             ▼                   │
│  ┌───────────────┐        ┌─────┐      ┌──────┐      ┌─────┐                │
│  │  AUTO-ENROLL  │───────→│SEARCH│────→│LOCKED│─────→│LOST │                │
│  └───────────────┘        └─────┘      └──────┘      └─────┘                │
│   🔊 enroll.wav(2x)     🔊 run.wav(2x)  🔇 Stop    (grace period)            │
│                                                                             │
│                                                                             │
│  t=49s (grace hết)       t=55s         t=57s                                │
│   │                        │             │                                  │
│   ▼                        ▼             ▼                                  │
│  ┌───────────┐          ┌─────┐      ┌──────┐                               │
│  │ SEARCHING │─────────→│LOCKED│     │LOCKED│                               │
│  └───────────┘          └─────┘      └──────┘                               │
│   🔊 lost.wav(LOOP)      🔇 Stop       ...                                   │
│   (phát liên tục)        lost sound                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 Thresholds

| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| $\tau_{accept}$ | 0.75 | Ngưỡng similarity để LOCK target |
| $\tau_{reject}$ | 0.6 | Ngưỡng similarity để mất target |
| `grace_period` | 2.0s | Thời gian chờ trong LOST trước khi quay lại SEARCHING |
| `enrollment_timeout` | 30s | Thời gian tối đa cho enrollment |
| `enrollment_samples` | 100 | Số mẫu tối đa cần thu thập |

### 6.7 Target Selection Algorithm (Code)

```python
def _find_best_track_by_reid(self, confirmed_tracks):
    best_track, best_score = None, -1.0
    
    for track in confirmed_tracks:
        # Cosine similarity với target feature
        score = np.dot(track.get_feature(), self.target_feature)
        
        if score > best_score:
            best_score = score
            best_track = track
    
    if best_score > τ_accept:
        return best_track
    return None
```

---

## 7. Module 6: Online Adaptation

### 7.1 Auto-Enrollment

**Thuật toán:**
1. Thu thập samples trong `timeout` giây (hoặc đến `target_samples`)
2. Tính centroid bằng EMA:
$$\mathbf{f}_{centroid}^{(t)} = 0.9 \cdot \mathbf{f}_{centroid}^{(t-1)} + 0.1 \cdot \mathbf{f}_{new}$$
3. Lưu làm anchor: $\mathbf{f}^* = \mathbf{f}_{centroid}$

### 7.2 Anchor-Based Model Update

**Vấn đề Model Drift:**
- Model bị "trôi" dần khỏi target gốc nếu update liên tục
- Giải pháp: Giữ **60% anchor** trong mỗi lần update

**Công thức Update:**
$$\mathbf{f}_{new} = w_{anchor} \cdot \mathbf{f}^* + w_{current} \cdot \mathbf{f}_{current} + w_{sample} \cdot \mathbf{f}_{sample}$$

Với:
- $w_{anchor} = 0.6$ (anchor weight - KHÔNG ĐỔI)
- $w_{current} = 0.3$ (current model)
- $w_{sample} = 0.1$ (new sample)

**Điều kiện Update:**
```python
if (similarity > τ_reject AND 
    similarity < 0.99 AND  # Diversity check
    time_since_last_update > 1.0s):
    adaptive_model_update()
```

### 7.3 Occlusion Detection

Phát hiện target bị che khuất bằng depth:

$$\text{occluded} = \begin{cases} 
\text{True} & \text{if } d_{current} < d_{last} - \tau_{occ} \\
\text{False} & \text{otherwise}
\end{cases}$$

Với $\tau_{occ} = 0.5m$ (occlusion threshold).

---

## 8. Module 7: Robot Control

### 8.1 Control Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ROBOT CONTROL LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Target Box (x1,y1,x2,y2)                                   │
│         │                                                   │
│         ├──→ Heading Error (pixels) ──→ Angular Velocity   │
│         │                                                   │
│         └──→ Depth (meters) ──→ Linear Velocity            │
│                                                             │
│  Output: Twist(linear.x, angular.z)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Heading Control (Angular Velocity)

**Error calculation:**
$$e_x = c_x - \frac{W}{2}$$

Với $c_x = \frac{x_1 + x_2}{2}$ là tâm target.

**Deadband:**
$$e_{eff} = \begin{cases} 
0 & \text{if } |e_x| \leq \delta_{dead} \\
\text{sign}(e_x) \cdot (|e_x| - \delta_{dead}) & \text{otherwise}
\end{cases}$$

**P-Control:**
$$\omega_z = \text{clamp}(-K_x \cdot e_{eff}, -\omega_{max}, +\omega_{max})$$

**Parameters:**
- $\delta_{dead} = 40$ px (deadband)
- $K_x = 0.00025$ (proportional gain)
- $\omega_{max} = 0.25$ rad/s

### 8.3 Distance Control (Linear Velocity)

**Depth EMA Filter:**
$$\hat{d}_t = \alpha \cdot d_{raw} + (1-\alpha) \cdot \hat{d}_{t-1}$$

Với $\alpha = 0.3$ (smoothing factor).

**Distance Error:**
$$e_d = \hat{d}_t - d_{target}$$

**P-Control (forward only):**
$$v_x = \begin{cases} 
\text{clamp}(K_d \cdot e_d, 0, v_{max}) & \text{if } e_d > 0 \text{ AND centered} \\
0 & \text{otherwise}
\end{cases}$$

**Parameters:**
- $d_{target} = 2.0$ m (desired distance)
- $K_d = 0.6$ (proportional gain)
- $v_{max} = 0.3$ m/s

### 8.4 Center-First Strategy

Robot ưu tiên căn giữa trước khi tiến:
```python
if center_first:
    if not is_centered:
        vx = 0  # Chỉ xoay, không tiến
```

---

## 9. Module 8: Audio Feedback System (Human-Robot Interaction)

### 9.1 Tổng Quan

Hệ thống sử dụng **phản hồi âm thanh** để giao tiếp với người dùng, giúp người được theo dõi biết trạng thái của robot mà không cần nhìn màn hình.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO FEEDBACK SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │ enroll_viet │     │  run_viet   │     │ lost_target_viet│   │
│  │    .wav     │     │    .wav     │     │      .wav       │   │
│  └──────┬──────┘     └──────┬──────┘     └────────┬────────┘   │
│         │                   │                     │             │
│         ↓                   ↓                     ↓             │
│   AUTO-ENROLL          SEARCHING              SEARCHING        │
│   (play 2x)            (play 2x)              (loop until      │
│                                                re-acquired)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Audio Files

| File | Nội dung | Khi nào phát |
|------|----------|--------------|
| `enroll_viet.wav` | "Mời bạn quay lưng lại và đứng trước camera để lấy mẫu" | Bắt đầu AUTO-ENROLL |
| `run_viet.wav` | "Bắt đầu theo dõi, bạn có thể di chuyển" | Sau khi enrollment hoàn thành |
| `lost_target_viet.wav` | "Mất target rồi, vui lòng quay lại để lấy mẫu" | Khi vào SEARCHING (loop) |

### 9.3 State-Audio Synchronization

```
┌────────────────────────────────────────────────────────────────────┐
│                    STATE-AUDIO STATE MACHINE                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐                                                   │
│  │ AUTO-ENROLL │ ──→ 🔊 enroll_viet.wav (2x)                       │
│  └──────┬──────┘     (play once, không lặp)                        │
│         │ enrollment done                                          │
│         ↓                                                          │
│  ┌─────────────┐                                                   │
│  │  SEARCHING  │ ──→ 🔊 run_viet.wav (2x)                          │
│  └──────┬──────┘     (play once khi vừa enroll xong)               │
│         │ target found                                             │
│         ↓                                                          │
│  ┌─────────────┐                                                   │
│  │   LOCKED    │ ──→ 🔇 Stop lost sound nếu đang phát              │
│  └──────┬──────┘                                                   │
│         │ target lost + grace_period expired                       │
│         ↓                                                          │
│  ┌─────────────┐                                                   │
│  │  SEARCHING  │ ──→ 🔊 lost_target_viet.wav (LOOP)                │
│  │  (from LOST)│     (phát liên tục đến khi re-acquire)            │
│  └─────────────┘                                                   │
│         │ target re-acquired                                       │
│         ↓                                                          │
│  ┌─────────────┐                                                   │
│  │   LOCKED    │ ──→ 🔇 Stop lost sound loop                       │
│  └─────────────┘                                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 9.4 Implementation Details

**9.4.1 One-time Audio Playback (Non-blocking)**

```python
# Phát 2 lần, không chờ (background process)
os.system(f"(aplay {enroll_sound_file}; aplay {enroll_sound_file}) &")
```

**9.4.2 Lost Sound Loop (Threading)**

Sử dụng **threading** để phát âm thanh liên tục mà không block main loop:

```python
class PersonDetector:
    def __init__(self):
        self.lost_sound_thread = None
        self.stop_lost_sound_event = threading.Event()
    
    def _lost_sound_loop(self):
        """Thread function: phát lost sound liên tục."""
        while not self.stop_lost_sound_event.is_set():
            if os.path.exists(self.sound_file):
                os.system(f"aplay {self.sound_file}")
            time.sleep(0.5)  # Delay giữa các lần phát
    
    def start_lost_sound_loop(self):
        """Bắt đầu phát lost sound."""
        if self.lost_sound_thread is not None:
            return  # Đang phát rồi
        
        self.stop_lost_sound_event.clear()
        self.lost_sound_thread = threading.Thread(
            target=self._lost_sound_loop, 
            daemon=True
        )
        self.lost_sound_thread.start()
    
    def stop_lost_sound_loop(self):
        """Dừng phát lost sound."""
        self.stop_lost_sound_event.set()
        self.lost_sound_thread.join(timeout=2.0)
        self.lost_sound_thread = None
```

### 9.5 Audio Trigger Logic

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Enroll Audio** | `state == AUTO-ENROLL` AND `enroll_audio_played == False` | Play 2x, set flag |
| **Run Audio** | `enrollment done` AND `run_audio_played == False` | Play 2x, set flag |
| **Lost Audio Start** | `state: LOST → SEARCHING` (grace_period expired) | Start loop thread |
| **Lost Audio Stop** | `state: SEARCHING → LOCKED` OR `state: LOST → LOCKED` | Stop loop thread |

### 9.6 Flags để Tránh Phát Lặp

```python
# Đảm bảo mỗi audio chỉ phát 1 lần
self.enroll_audio_played = False  # Reset khi khởi động
self.run_audio_played = False     # Reset khi khởi động

# Trong on_image callback:
if not self.enroll_audio_played:
    os.system(f"(aplay {enroll_sound}; aplay {enroll_sound}) &")
    self.enroll_audio_played = True  # Đánh dấu đã phát
```

### 9.7 Ưu Điểm của Audio Feedback

| Ưu điểm | Giải thích |
|---------|------------|
| **Hands-free** | Người dùng không cần nhìn màn hình |
| **Accessibility** | Hữu ích cho người khiếm thị |
| **Non-blocking** | Sử dụng background process/thread |
| **State-aware** | Âm thanh phản ánh đúng trạng thái hệ thống |
| **Loop for attention** | Lost sound lặp liên tục để thu hút sự chú ý |

---

## 10. So Sánh với DeepSORT Gốc

| Thành phần | DeepSORT Gốc | Hệ thống này |
|------------|--------------|--------------|
| **Detection** | Faster R-CNN | MobileNet-SSD (nhẹ hơn) |
| **ReID Feature** | CNN embedding | MobileNetV2 + HSV + Depth |
| **Matching** | Cascade + Hungarian | Tương tự |
| **Kalman** | 8D state | Tương tự |
| **Target Selection** | Multi-target | **Single-target với State Machine** |
| **Model Update** | Không có | **Anchor-based Adaptation** |
| **Occlusion** | Không xử lý | **Depth-based Detection** |
| **Audio Feedback** | Không có | **Voice guidance với threading** |
| **Control** | Không có | **P-Control với EMA** |

---

## 11. Complexity Analysis

### 11.1 Time Complexity

| Module | Complexity | Notes |
|--------|------------|-------|
| Detection | $O(1)$ | Fixed input size 300×300 |
| Feature Extraction | $O(N)$ | $N$ = số detections |
| Data Association | $O(M \cdot N)$ | $M$ = số tracks, Hungarian $O(n^3)$ |
| Kalman Filter | $O(M)$ | Mỗi track update $O(1)$ |
| Audio Feedback | $O(1)$ | Background thread, non-blocking |
| Total per frame | $O(M \cdot N)$ | Với $M, N$ nhỏ (2-5 người) |

### 11.2 Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| Feature vector | 1584 × 4 bytes | ~6.2 KB/detection |
| Track history | 30 × 1584 × 4 bytes | ~185 KB/track |
| Kalman state | 8 × 8 bytes | 64 bytes/track |
| Audio files | ~568 KB total | 3 WAV files |

---

## 12. Kết Luận

Hệ thống kết hợp nhiều kỹ thuật từ computer vision, robotics và human-robot interaction:

1. **Detection**: MobileNet-SSD cho real-time performance
2. **Feature Extraction**: Multi-modal (shape + color + depth) cho robustness
3. **Data Association**: Hungarian matching với Kalman gating
4. **Motion Prediction**: Kalman Filter cho smooth tracking
5. **Target Management**: State machine cho single-target focus
6. **Online Adaptation**: Anchor-based update chống model drift
7. **Robot Control**: P-control với depth feedback
8. **Audio Feedback**: Voice guidance cho human-robot interaction

**Contributions so với DeepSORT gốc:**
- Tích hợp depth feature từ RGB-D camera
- Anchor-based model update chống drift
- Single-target state machine
- Dynamic color weight cho điều kiện ánh sáng thay đổi
- Audio feedback system với threading cho voice guidance


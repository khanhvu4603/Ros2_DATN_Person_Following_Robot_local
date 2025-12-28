# So Sánh Thuật Toán Tracking

So sánh chi tiết giữa thuật toán tracking hiện tại (Anchor-ReID Single Tracker) với DeepSORT và ByteTrack.

---

## 📊 Bảng So Sánh Tổng Quan

| Tiêu chí | **DeepSORT** | **ByteTrack** | **Thuật toán hiện tại** |
|----------|--------------|---------------|-------------------------|
| **Mục đích** | Multi-Object Tracking | Multi-Object Tracking | **Single-Target Tracking** |
| **Số targets** | Nhiều | Nhiều | **1 duy nhất** |
| **Kalman Filter** | ✅ 8-dim state | ✅ 8-dim state | ✅ 8-dim state |
| **Hungarian Algorithm** | ✅ Có | ✅ Có | ❌ **Không** |
| **Re-ID features** | ✅ CNN features | ❌ Không | ✅ **MobileNetV2 + HSV + Depth** |
| **Low-score detections** | ❌ Bỏ qua | ✅ Sử dụng (BYTE) | ❌ Bỏ qua |
| **Identity verification** | Cascade matching | IoU-based | **Anchor-based ReID** |
| **Occlusion handling** | Kalman prediction | Kalman + low-score | **Depth gating + Kalman** |

---

## 1. DeepSORT

### Pipeline
```
Detection → Kalman Predict → Cascade Matching → Hungarian Assignment → Update Tracks
```

### Đặc điểm
- **Cascade Matching**: Ưu tiên match tracks mới hơn (time_since_update nhỏ)
- **Cost Matrix**: λ × IoU_cost + (1-λ) × Appearance_cost
- **Track lifecycle**: Tentative → Confirmed → Deleted
- **Re-ID**: CNN embedding (thường dùng OSNet, ResNet)

### Ưu điểm
- Xử lý tốt nhiều người
- Re-ID giúp giảm ID switch sau occlusion
- Cascade matching ưu tiên tracks gần đây

### Nhược điểm
- ❌ Có thể switch ID khi người giống nhau
- ❌ Hungarian matching phức tạp cho 1 target
- ❌ Feature có thể drift qua thời gian

---

## 2. ByteTrack

### Pipeline
```
Detection → Split High/Low score → Match High → Match Low → Update Tracks
```

### Đặc điểm
- **BYTE strategy**: Sử dụng cả low-score detections
- **Không dùng Re-ID**: Chỉ dùng IoU
- **Two-stage matching**: 
  - Stage 1: Match high-score detections với tracks
  - Stage 2: Match low-score detections với unmatched tracks

### Ưu điểm
- Nhanh (không có Re-ID inference)
- Robust với occlusion (dùng low-score detections)
- State-of-the-art trên MOT benchmark

### Nhược điểm
- ❌ Dễ switch ID (không có appearance)
- ❌ Không phân biệt được người giống nhau
- ❌ Cần tuning threshold cho high/low score

---

## 3. Thuật Toán Hiện Tại (Anchor-ReID Single Tracker)

### Pipeline
```
Detection → Depth Filter → IoU Match predicted box → ReID Verify (anchor) → EMA Smooth → Update
```

### Đặc điểm
- **Single-target only**: Không có multi-track management
- **Anchor-based ReID**: So sánh với mẫu gốc (không drift)
- **Depth gating**: Lọc người đứng gần hơn target
- **EMA smoothing**: Làm mượt box position
- **No Hungarian**: Không cần vì chỉ 1 target

### Ưu điểm
- ✅ Chống switch ID tốt (anchor-based)
- ✅ Đơn giản, nhẹ CPU
- ✅ Depth-aware (phát hiện occlusion)
- ✅ Không drift (60% anchor weight)

### Nhược điểm
- ❌ Chỉ track được 1 người
- ❌ Phụ thuộc chất lượng enrollment
- ❌ Cần camera depth

---

## 📈 So Sánh Chi Tiết Các Thành Phần

### A. Kalman Filter

| | DeepSORT | ByteTrack | Hiện tại |
|-|----------|-----------|----------|
| State | [x, y, a, h, vx, vy, va, vh] | [x, y, a, h, vx, vy, va, vh] | [x, y, a, h, vx, vy, va, vh] |
| Dùng để | Predict + Smooth | Predict + Smooth | **Chỉ Predict** (khi mất detection) |
| Position output | Kalman smoothed | Kalman smoothed | **Raw detection + EMA** |

### B. Data Association

| | DeepSORT | ByteTrack | Hiện tại |
|-|----------|-----------|----------|
| Algorithm | Hungarian | Hungarian (2-stage) | **Direct IoU + ReID verify** |
| Cost | IoU + Appearance | IoU only | IoU (filter) + Appearance (verify) |
| Complexity | O(n³) | O(n³) | **O(n)** |

### C. Appearance Features

| | DeepSORT | ByteTrack | Hiện tại |
|-|----------|-----------|----------|
| CNN | ResNet/OSNet | ❌ Không | MobileNetV2 |
| Color | ❌ Không | ❌ Không | ✅ HSV histogram |
| Depth | ❌ Không | ❌ Không | ✅ Depth feature |
| Update | EMA update | N/A | **Anchor-based** (60% gốc) |

### D. Occlusion Handling

| | DeepSORT | ByteTrack | Hiện tại |
|-|----------|-----------|----------|
| Detection | Kalman predict | Low-score detections | **Depth jump detection** |
| Recovery | Re-match after occlusion | Re-match với low-score | **Anchor-based re-acquire** |
| Depth-aware | ❌ | ❌ | ✅ |

---

## 🎯 Khi Nào Dùng Thuật Toán Nào?

| Use Case | Recommended |
|----------|-------------|
| Track nhiều người, camera 2D | ByteTrack |
| Track nhiều người, cần phân biệt ID | DeepSORT |
| **Robot theo 1 người, camera RGB-D** | **Thuật toán hiện tại** |
| Real-time trên edge device | ByteTrack hoặc Hiện tại |

---

## 📝 Tên Gọi Cho Thuật Toán Hiện Tại

Có thể gọi thuật toán này là:

> **"Anchor-ReID Single Target Tracker with Depth Gating"**

Hoặc ngắn gọn: **"Anchor-STT"** (Anchor Single-Target Tracker)

### Đặc trưng chính:
1. **Single-Target** - Chỉ track 1 người
2. **Anchor-based** - So sánh với mẫu gốc, chống drift
3. **Depth-aware** - Sử dụng depth để lọc và phát hiện occlusion
4. **No Hungarian** - Không cần matching algorithm phức tạp

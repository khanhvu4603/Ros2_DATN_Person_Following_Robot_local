# 📊 So Sánh 3 Phiên Bản Feature Extraction

> **Mục đích**: So sánh hiệu quả tracking với các tổ hợp features khác nhau  
> **Ngày**: 17/12/2024  
> **Video test**: Không có ground truth bounding box

---

## 1. Tổng Quan 3 Phiên Bản

| Version | Features | Dimension | File |
|---------|----------|-----------|------|
| **V1** | MobileNetV2 (Shape) | 1280 | `tracker_shape_only.py` |
| **V2** | Shape + Depth | 1536 | `tracker_shape_depth.py` |
| **V3** | Shape + Depth + HSV | 1584 | `tracker_full.py` |

### Feature Vector Composition

```
V1: [MobileNetV2] = 1280-dim
     └── Shape/Pose features

V2: [MobileNetV2, Depth] = 1280 + 256 = 1536-dim
     └── Shape   └── 16×16 depth map

V3: [MobileNetV2, Depth, HSV] = 1280 + 256 + 48 = 1584-dim
     └── Shape   └── Depth  └── Color histogram
```

---

## 2. Metrics Đánh Giá (Không Cần Ground Truth)

### 2.1 Tracking Stability Metrics

| Metric | Ký hiệu | Mô tả | Tốt hơn |
|--------|---------|-------|---------|
| **Track Fragmentations** | `TF` | Số lần LOCKED → LOST → LOCKED | Thấp ↓ |
| **Lost Rate** | `LR` | % thời gian ở LOST/SEARCHING | Thấp ↓ |
| **Longest Track Duration** | `LTD` | Thời gian LOCKED liên tục dài nhất | Cao ↑ |
| **Average Track Duration** | `ATD` | Trung bình thời gian LOCKED mỗi lần | Cao ↑ |

### 2.2 Re-Identification Metrics

| Metric | Ký hiệu | Mô tả | Tốt hơn |
|--------|---------|-------|---------|
| **Re-ID Success Rate** | `RSR` | % lần tìm lại được sau LOST | Cao ↑ |
| **Time to Re-acquire** | `TTR` | Thời gian trung bình từ LOST → LOCKED | Thấp ↓ |
| **False Re-ID Rate** | `FRR` | % lần lock nhầm người khác (manual check) | Thấp ↓ |

### 2.3 Similarity Metrics

| Metric | Ký hiệu | Mô tả | Tốt hơn |
|--------|---------|-------|---------|
| **Mean Similarity** | `μ_sim` | Similarity trung bình khi LOCKED | Cao ↑ |
| **Similarity Std** | `σ_sim` | Độ dao động similarity | Thấp ↓ |
| **Min Similarity** | `min_sim` | Similarity thấp nhất vẫn LOCKED | Cao ↑ |
| **Similarity Margin** | `SM` | `μ_sim - reject_threshold` | Cao ↑ |

### 2.4 Performance Metrics

| Metric | Ký hiệu | Mô tả |
|--------|---------|-------|
| **FPS** | `FPS` | Frames per second |
| **Processing Time** | `PT` | ms/frame |
| **Feature Extraction Time** | `FET` | ms/feature |

---

## 3. Bảng So Sánh (Template)

Sau khi chạy test trên video, điền kết quả vào bảng:

### 3.1 Tracking Stability

| Metric | V1 (Shape) | V2 (+Depth) | V3 (+HSV) | Best |
|--------|------------|-------------|-----------|------|
| Track Fragmentations | | | | |
| Lost Rate (%) | | | | |
| Longest Track (s) | | | | |
| Average Track (s) | | | | |

### 3.2 Re-Identification

| Metric | V1 (Shape) | V2 (+Depth) | V3 (+HSV) | Best |
|--------|------------|-------------|-----------|------|
| Re-ID Success Rate (%) | | | | |
| Time to Re-acquire (s) | | | | |
| False Re-ID (count) | | | | |

### 3.3 Similarity Statistics

| Metric | V1 (Shape) | V2 (+Depth) | V3 (+HSV) | Best |
|--------|------------|-------------|-----------|------|
| Mean Similarity | | | | |
| Std Similarity | | | | |
| Min Similarity | | | | |
| Similarity Margin | | | | |

### 3.4 Performance

| Metric | V1 (Shape) | V2 (+Depth) | V3 (+HSV) | Best |
|--------|------------|-------------|-----------|------|
| FPS | | | | |
| Processing Time (ms) | | | | |

---

## 4. Log Format

### 4.1 CSV Log (mỗi frame)

```csv
frame_id,timestamp,state,similarity,bbox_x,bbox_y,bbox_w,bbox_h,depth_m,proc_time_ms
```

**Ví dụ**:
```csv
1,0.033,LOCKED,0.823,150,100,80,200,2.35,45.2
2,0.066,LOCKED,0.815,152,101,78,198,2.32,43.8
3,0.100,LOST,0.450,155,102,75,195,1.20,42.1
```

### 4.2 Event Log (state changes)

```csv
timestamp,from_state,to_state,trigger,similarity
```

**Ví dụ**:
```csv
0.100,LOCKED,LOST,occlusion,0.82
2.100,LOST,SEARCHING,grace_expired,0.45
5.300,SEARCHING,LOCKED,reid_match,0.78
```

### 4.3 Summary Statistics (kết thúc video)

```
========== TRACKING SUMMARY ==========
Version: V3 (Shape + Depth + HSV)
Video: test_video_01.mp4
Duration: 120.0 seconds
Total Frames: 3600

--- State Distribution ---
LOCKED:     92.5% (3330 frames)
LOST:        3.2% (115 frames)  
SEARCHING:   4.3% (155 frames)

--- Tracking Stability ---
Track Fragmentations: 4
Longest Track: 45.2 seconds
Average Track: 23.1 seconds

--- Re-ID Performance ---
Lost Events: 5
Re-acquired: 4
Re-ID Success Rate: 80.0%
Avg Time to Re-acquire: 1.8 seconds

--- Similarity Statistics ---
Mean: 0.812 ± 0.042
Min:  0.621
Max:  0.903
Margin: 0.212 (above reject_thr=0.60)

--- Performance ---
FPS: 24.3
Avg Processing Time: 41.2 ms
==========================================
```

---

## 5. Công Thức Tính

### Track Fragmentation (TF)
```
TF = Số lần (LOCKED → LOST → LOCKED)
```

### Lost Rate (LR)
```
LR = (frames_in_LOST + frames_in_SEARCHING) / total_frames × 100%
```

### Re-ID Success Rate (RSR)
```
RSR = times_reacquired / times_lost × 100%
```

### Similarity Margin (SM)
```
SM = mean_similarity - reject_threshold
   = μ_sim - 0.60
```

---

## 6. Kịch Bản Test Đề Xuất

### Scenario 1: Normal Tracking
- Target đi thẳng, không che khuất
- Đánh giá: Stability, Similarity

### Scenario 2: Occlusion
- Người khác đi ngang qua che target
- Đánh giá: Re-ID, Time to Re-acquire

### Scenario 3: Appearance Change
- Target quay người (lưng, bên hông)
- Đánh giá: Similarity variance

### Scenario 4: Distance Change
- Target đi ra xa rồi lại gần
- Đánh giá: Depth feature effectiveness

### Scenario 5: Lighting Change
- Ánh sáng thay đổi (vào/ra khỏi bóng)
- Đánh giá: HSV robustness

---

## 7. Dự Đoán Kết Quả

### Hypothesis

| Metric | V1 vs V2 | V2 vs V3 |
|--------|----------|----------|
| Similarity | V2 > V1 | V3 ≈ V2 |
| Occlusion handling | V2 >> V1 | V3 ≈ V2 |
| Distance change | V2 >> V1 | V3 ≈ V2 |
| Lighting change | V2 ≈ V1 | V3 > V2 |
| Different person | V2 > V1 | V3 >> V2 |
| Speed (FPS) | V1 > V2 | V2 > V3 |

### Kỳ Vọng

- **V1 (Shape only)**: Nhanh nhất, nhưng dễ nhầm người có dáng giống
- **V2 (+ Depth)**: Tốt hơn khi có occlusion, phân biệt khoảng cách
- **V3 (+ HSV)**: Tốt nhất để phân biệt nhiều người, robust với lighting

---

## 8. Kết Luận (Điền sau khi test)

### Best Overall: `V?`

### Recommendation:
- Nếu cần **tốc độ cao**: V1
- Nếu có **nhiều người, occlusion**: V2 hoặc V3
- Nếu môi trường **ánh sáng thay đổi nhiều**: V3

---

*Template created: 17/12/2024*

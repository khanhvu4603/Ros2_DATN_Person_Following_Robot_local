# VFH+ (Vector Field Histogram Plus) cho Robot Mecanum

## Mục lục
1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [Lý thuyết VFH cơ bản](#2-lý-thuyết-vfh-cơ-bản)
3. [VFH+ cải tiến](#3-vfh-cải-tiến)
4. [Tích hợp Potential Field](#4-tích-hợp-potential-field)
5. [Holonomic Extension cho Mecanum](#5-holonomic-extension-cho-mecanum)
6. [Kiến trúc tích hợp vào Project](#6-kiến-trúc-tích-hợp-vào-project)
7. [Thuật toán chi tiết](#7-thuật-toán-chi-tiết)
8. [Công thức toán học](#8-công-thức-toán-học)
9. [Tham số và tuning](#9-tham-số-và-tuning)
10. [Xử lý edge cases](#10-xử-lý-edge-cases)
11. [Kế hoạch triển khai](#11-kế-hoạch-triển-khai)

---

## 1. Giới thiệu tổng quan

### 1.1 Bài toán
Robot Mecanum cần:
- **Bám theo target** (người) được detect bởi camera + DeepSORT
- **Tránh vật cản** được detect bởi Lidar 2D 360°
- **Di chuyển mượt mà** không giật, không xung đột lệnh

### 1.2 Tại sao chọn VFH+?

| Thuật toán | Target động | Vật cản động | Local minima | Holonomic | Real-time |
|------------|-------------|--------------|--------------|-----------|-----------|
| A* | ❌ | ❌ | ✅ | ✅ | ⚠️ |
| RRT | ❌ | ❌ | ✅ | ✅ | ❌ |
| APF | ✅ | ✅ | ❌ | ✅ | ✅ |
| VFH | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| **VFH+** | ✅ | ✅ | ✅ | ✅ | ✅ |

### 1.3 VFH+ = VFH + APF + Holonomic

```
VFH+  =  VFH Core     (tìm đường không bị kẹt)
       + APF          (làm mượt trajectory)  
       + Holonomic    (tận dụng đi ngang của Mecanum)
```

---

## 2. Lý thuyết VFH cơ bản

### 2.1 Ý tưởng chính

Thay vì tính toán đường đi toàn cục, VFH:
1. **Quét môi trường 360°** bằng Lidar
2. **Tạo histogram** mật độ vật cản theo góc
3. **Tìm các "valley"** (khe hở) an toàn
4. **Chọn valley tốt nhất** gần hướng mục tiêu

### 2.2 Polar Histogram

```
Mật độ vật cản
  ▲
12│        ████
10│    ████████
 8│    ████████  ████
 6│    ████████  ████
 4│    ████████  ████    ██
 2│    ████████  ████    ██
 0│────████████──████────██──────────────► Góc (°)
   0°  45°  90° 135° 180° 225° 270° 315° 360°
          │              │
          │              └── Valley 2 (wide)
          └── Valley 1 (narrow)
```

### 2.3 Công thức Histogram

Chia 360° thành `n` sectors (ví dụ: 72 sectors × 5°/sector):

```
h[k] = Σ c(i)² × d(i)     với i thuộc sector k

Trong đó:
- c(i) = certainty value của obstacle i (0-1)
- d(i) = (d_max - d_obs) / d_max  (gần → cao, xa → thấp)
- h[k] = mật độ vật cản của sector k
```

### 2.4 Tìm Valley

```
threshold = mean(h) + α × std(h)

Valley = chuỗi sectors liên tiếp có h[k] < threshold
```

---

## 3. VFH+ cải tiến

### 3.1 Cải tiến so với VFH gốc

| VFH gốc | VFH+ |
|---------|------|
| Chỉ xét angle | Xét cả turning cost |
| Binary threshold | Smooth cost function |
| Không xét robot size | Wide/Narrow valley |
| Single stage | Multi-stage selection |

### 3.2 Three-Stage Selection

```
Stage 1: Binary Histogram
├── h[k] > threshold → blocked
└── h[k] ≤ threshold → free

Stage 2: Masked Histogram  
├── Áp dụng robot radius
└── Wide vs Narrow valleys

Stage 3: Direction Selection
├── Minimize: goal_cost + turn_cost + previous_cost
└── Smooth transition giữa các frame
```

### 3.3 Valley Classification

```
Wide Valley:   θ_width > 2 × (s_max + robot_radius)
Narrow Valley: θ_width ≤ 2 × (s_max + robot_radius)

Trong đó:
- θ_width = góc valley
- s_max = sector angular resolution
- robot_radius = bán kính tối thiểu chứa robot
```

---

## 4. Tích hợp Potential Field

### 4.1 Tại sao cần APF?

VFH chọn direction RỜI RẠC (theo sector), có thể gây giật khi valley thay đổi.
APF làm MƯỢT direction bằng cách blend với lực hấp dẫn/đẩy.

### 4.2 Attractive Force (Target)

```
F_att = K_att × (P_target - P_robot) / ‖P_target - P_robot‖

Trong đó:
- K_att = hệ số hấp dẫn (default: 1.0)
- P_target = vị trí target (x, y) trong robot frame
- P_robot = (0, 0)
```

### 4.3 Repulsive Force (Obstacles)

```
            ┌ K_rep × (1/d - 1/d_max)² × n̂   nếu d ≤ d_max
F_rep(i) = ┤
            └ 0                                nếu d > d_max

Trong đó:
- K_rep = hệ số đẩy (default: 0.5)
- d = khoảng cách đến obstacle i
- d_max = ngưỡng ảnh hưởng (default: 1.5m)
- n̂ = vector đơn vị từ obstacle hướng robot
```

### 4.4 Blending VFH + APF

```
θ_vfh = hướng từ VFH valley selection
θ_apf = atan2(F_att + Σ F_rep)

θ_final = β × θ_vfh + (1 - β) × θ_apf

Trong đó:
- β = blending factor (default: 0.7)
- β cao → ưu tiên VFH (tránh vật cản)
- β thấp → ưu tiên APF (mượt mà)
```

---

## 5. Holonomic Extension cho Mecanum

### 5.1 Lợi thế Mecanum

Robot Mecanum có thể di chuyển **omni-directional**:
- **vx**: Tiến/lùi
- **vy**: Đi ngang trái/phải
- **wz**: Xoay tại chỗ

→ Có thể vừa tiến về target vừa đi ngang tránh vật cản!

### 5.2 Decomposition Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    θ_desired (từ VFH+)                  │
│                         │                               │
│            ┌────────────┼────────────┐                  │
│            ▼            ▼            ▼                  │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│    │ vx Component │ │ vy Component │ │ wz Component │   │
│    │ (Forward)    │ │ (Lateral)    │ │ (Heading)    │   │
│    └──────────────┘ └──────────────┘ └──────────────┘   │
│            │            │            │                  │
│            ▼            ▼            ▼                  │
│         cos(θ)       sin(θ)    error_heading            │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Velocity Decomposition

```python
# θ_desired: hướng di chuyển tối ưu (từ VFH+)
# θ_current: hướng robot đang nhìn (0° = phía trước)
# θ_target:  hướng đến target (từ camera)

# 1. Forward velocity (vx) - Tiến về target
vx = K_d × distance_to_target × cos(θ_desired - θ_current)
vx = clamp(vx, 0, vx_max)  # Không lùi

# 2. Lateral velocity (vy) - Đi ngang tránh vật cản  
vy = K_lat × sin(θ_desired - θ_current)
vy = clamp(vy, -vy_max, vy_max)

# 3. Angular velocity (wz) - Xoay heading về target
heading_error = θ_target - θ_current
wz = K_w × heading_error
wz = clamp(wz, -wz_max, wz_max)
```

### 5.4 Ví dụ minh họa

```
Tình huống: Target ở phía trước (θ_target = 0°)
            Vật cản ở trái (VFH chọn θ_desired = 30°)

Kết quả:
- vx = K × cos(30°) = 0.87 × K   → Tiến về trước (giảm nhẹ)
- vy = K × sin(30°) = 0.50 × K   → Đi sang PHẢI
- wz = K × 0° = 0                → Không xoay (đã nhìn target)

→ Robot đi CHÉO: vừa tiến vừa sang phải để tránh vật cản
```

---

## 6. Kiến trúc tích hợp vào Project

### 6.1 Kiến trúc hiện tại

```
┌─────────────────┐     ┌─────────────────┐
│ Person Detector │     │ Lidar Processor │
│ /cmd_vel_person │     │/cmd_vel_emergency│
│   (vx, wz)      │     │    (vy, wz)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Velocity   │
              │   Arbiter   │  ← MERGE đơn giản
              └──────┬──────┘
                     │
                Robot Motors
```

**Vấn đề**: Xung đột wz, không tối ưu, có thể bị kẹt.

### 6.2 Kiến trúc đề xuất với VFH+

```
┌─────────────────┐     ┌─────────────────┐
│ Person Detector │     │   Lidar Scan    │
│  θ_target       │     │   360° ranges   │
│  d_target       │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  VFH+ Node  │  ← NEW: Unified planner
              │             │
              │ 1. Build histogram
              │ 2. Find valleys
              │ 3. Score + select
              │ 4. APF smoothing
              │ 5. Holonomic decomp
              └──────┬──────┘
                     │
              /cmd_vel_vfh
              (vx, vy, wz optimal)
                     │
              ┌──────▼──────┐
              │  Velocity   │
              │   Arbiter   │  ← Simplified: just safety check
              └──────┬──────┘
                     │
                Robot Motors
```

### 6.3 ROS2 Topics

| Topic | Type | Publisher | Subscriber | Mô tả |
|-------|------|-----------|------------|-------|
| `/scan` | LaserScan | rplidar_node | vfh_planner | Lidar 360° |
| `/person_detector/target_polar` | Custom | person_detector | vfh_planner | (θ, d) của target |
| `/cmd_vel_vfh` | Twist | vfh_planner | velocity_arbiter | Velocity tối ưu |
| `/vfh_debug` | Image | vfh_planner | rviz/debug | Visualization |

### 6.4 File cần tạo/sửa

```
src/mecanum_control/mecanum_control/
├── vfh_planner.py          # NEW: VFH+ node chính
├── vfh/
│   ├── __init__.py         # NEW
│   ├── histogram.py        # NEW: Build polar histogram
│   ├── valley_finder.py    # NEW: Find and score valleys
│   ├── apf_smoother.py     # NEW: APF smoothing
│   └── holonomic_decomp.py # NEW: Velocity decomposition
├── person_detector.py      # MODIFY: Publish target polar
├── velocity_arbiter.py     # MODIFY: Subscribe /cmd_vel_vfh
└── lidar_processor.py      # KEEP or REMOVE (replaced by VFH+)
```

---

## 7. Thuật toán chi tiết

### 7.1 Main Loop (vfh_planner.py)

```
FUNCTION vfh_plan(lidar_scan, target_polar):
    
    # Step 1: Build Polar Histogram
    histogram = build_histogram(lidar_scan)
    
    # Step 2: Apply robot radius mask
    masked_hist = apply_mask(histogram, robot_radius)
    
    # Step 3: Find valleys
    valleys = find_valleys(masked_hist, threshold)
    
    # Step 4: Score each valley
    IF valleys is empty:
        RETURN emergency_stop()
    
    FOR each valley v:
        v.score = goal_cost(v, target_polar) 
                + turn_cost(v, current_heading)
                + width_cost(v)
                + smooth_cost(v, previous_direction)
    
    # Step 5: Select best valley
    best_valley = argmin(valley.score)
    θ_vfh = center_of(best_valley)
    
    # Step 6: APF smoothing
    F_att = compute_attractive(target_polar)
    F_rep = compute_repulsive(lidar_scan)
    θ_apf = atan2(F_att + F_rep)
    
    θ_final = blend(θ_vfh, θ_apf, β=0.7)
    
    # Step 7: Holonomic decomposition
    vx, vy, wz = holonomic_decompose(θ_final, target_polar)
    
    # Step 8: Safety check
    vx, vy, wz = safety_clamp(vx, vy, wz, min_obstacle_dist)
    
    RETURN Twist(vx, vy, wz)
```

### 7.2 Build Histogram

```
FUNCTION build_histogram(scan, n_sectors=72):
    sector_size = 360° / n_sectors  # = 5°
    histogram = zeros(n_sectors)
    
    FOR each point (range, angle) in scan:
        IF range < range_min OR range > range_max:
            CONTINUE  # Invalid reading
        
        sector_idx = floor(angle / sector_size) MOD n_sectors
        
        # Certainty value (closer = higher)
        certainty = (range_max - range) / range_max
        
        histogram[sector_idx] += certainty²
    
    # Normalize
    histogram = histogram / max(histogram)
    
    RETURN histogram
```

### 7.3 Find Valleys

```
FUNCTION find_valleys(histogram, threshold_factor=0.5):
    threshold = mean(histogram) + threshold_factor × std(histogram)
    
    valleys = []
    in_valley = False
    start_idx = -1
    
    FOR k = 0 to len(histogram) + wrap_around:
        idx = k MOD len(histogram)
        
        IF histogram[idx] < threshold:
            IF NOT in_valley:
                in_valley = True
                start_idx = idx
        ELSE:
            IF in_valley:
                in_valley = False
                valleys.append(Valley(start_idx, idx-1))
    
    RETURN valleys
```

### 7.4 Score Valleys

```
FUNCTION score_valley(valley, target_angle, current_heading, prev_direction):
    θ_center = center_angle(valley)
    θ_width = valley.end - valley.start  # Wrapped
    
    # Cost components
    goal_cost = |θ_center - target_angle| / 180°           # [0, 1]
    turn_cost = |θ_center - current_heading| / 180°        # [0, 1]
    width_cost = 1.0 / (θ_width + 1)                       # Narrow = high cost
    smooth_cost = |θ_center - prev_direction| / 180°       # [0, 1]
    
    # Weighted sum
    total = w1 × goal_cost + w2 × turn_cost + w3 × width_cost + w4 × smooth_cost
    
    # Default weights
    # w1 = 1.0 (goal priority)
    # w2 = 0.3 (prefer less turning)
    # w3 = 0.5 (prefer wide valleys)
    # w4 = 0.2 (smooth trajectory)
    
    RETURN total
```

---

## 8. Công thức toán học

### 8.1 Polar Histogram

$$h_k = \sum_{i \in S_k} c_i^2 \cdot d_i$$

Trong đó:
- $S_k$ = tập hợp points trong sector $k$
- $c_i = \frac{r_{max} - r_i}{r_{max}}$ (certainty)
- $d_i = 1$ (simplified) hoặc distance factor

### 8.2 Valley Threshold

$$\tau = \mu(h) + \alpha \cdot \sigma(h)$$

Trong đó:
- $\mu(h)$ = mean của histogram
- $\sigma(h)$ = standard deviation
- $\alpha$ = tuning parameter (default: 0.5)

### 8.3 Attractive Force

$$\vec{F}_{att} = K_{att} \cdot \frac{\vec{p}_{target} - \vec{p}_{robot}}{||\vec{p}_{target} - \vec{p}_{robot}||}$$

### 8.4 Repulsive Force

$$\vec{F}_{rep}^{(i)} = 
\begin{cases}
K_{rep} \cdot \left(\frac{1}{d_i} - \frac{1}{d_{max}}\right)^2 \cdot \hat{n}_i & \text{if } d_i \leq d_{max} \\
0 & \text{otherwise}
\end{cases}$$

### 8.5 Direction Blending

$$\theta_{final} = \beta \cdot \theta_{VFH} + (1 - \beta) \cdot \theta_{APF}$$

### 8.6 Holonomic Velocities

$$v_x = K_d \cdot d_{target} \cdot \cos(\theta_{final} - \theta_{current})$$

$$v_y = K_{lat} \cdot \sin(\theta_{final} - \theta_{current})$$

$$\omega_z = K_\omega \cdot (\theta_{target} - \theta_{current})$$

---

## 9. Tham số và tuning

### 9.1 Histogram Parameters

| Parameter | Default | Range | Mô tả |
|-----------|---------|-------|-------|
| `n_sectors` | 72 | 36-180 | Số sectors (5° mỗi sector) |
| `range_min` | 0.1m | 0.05-0.2 | Bỏ qua readings quá gần |
| `range_max` | 5.0m | 3.0-10.0 | Bỏ qua readings quá xa |
| `threshold_factor` | 0.5 | 0.3-0.8 | Valley threshold |

### 9.2 Valley Selection Weights

| Parameter | Default | Range | Mô tả |
|-----------|---------|-------|-------|
| `w_goal` | 1.0 | 0.5-2.0 | Ưu tiên hướng target |
| `w_turn` | 0.3 | 0.1-0.5 | Giảm xoay |
| `w_width` | 0.5 | 0.2-1.0 | Ưu tiên valley rộng |
| `w_smooth` | 0.2 | 0.1-0.4 | Ổn định trajectory |

### 9.3 APF Parameters

| Parameter | Default | Range | Mô tả |
|-----------|---------|-------|-------|
| `K_att` | 1.0 | 0.5-2.0 | Lực hút target |
| `K_rep` | 0.5 | 0.2-1.0 | Lực đẩy obstacle |
| `d_influence` | 1.5m | 1.0-3.0 | Phạm vi ảnh hưởng repulsive |
| `β_blend` | 0.7 | 0.5-0.9 | VFH vs APF blend |

### 9.4 Velocity Parameters

| Parameter | Default | Range | Mô tả |
|-----------|---------|-------|-------|
| `K_d` | 0.5 | 0.3-1.0 | Gain distance → vx |
| `K_lat` | 0.3 | 0.2-0.5 | Gain lateral → vy |
| `K_omega` | 0.8 | 0.5-1.5 | Gain heading → wz |
| `vx_max` | 0.4 m/s | 0.2-0.6 | Max forward |
| `vy_max` | 0.3 m/s | 0.2-0.4 | Max lateral |
| `wz_max` | 0.5 rad/s | 0.3-0.8 | Max angular |

### 9.5 Tuning Guidelines

```
Nếu robot bị kẹt:
  → Giảm threshold_factor (nhiều valley hơn)
  → Tăng w_width (ưu tiên valley rộng)

Nếu robot đi giật:
  → Tăng w_smooth
  → Tăng β_blend (ưu tiên VFH)
  → Thêm EMA filter cho output

Nếu robot không bám kịp target:
  → Tăng K_d, vx_max
  → Giảm w_turn

Nếu robot đâm vật cản:
  → Tăng K_rep
  → Tăng d_influence
  → Giảm vx_max
```

---

## 10. Xử lý edge cases

### 10.1 Không có valley (bị vây)

```python
if len(valleys) == 0:
    # Option 1: Emergency stop
    return Twist(vx=0, vy=0, wz=0)
    
    # Option 2: Lùi lại
    return Twist(vx=-0.1, vy=0, wz=0)
    
    # Option 3: Xoay tại chỗ tìm gap
    return Twist(vx=0, vy=0, wz=0.3)
```

### 10.2 Mất target (camera không thấy)

```python
if target_polar is None:
    # Dùng last known position với Kalman predict
    target_polar = kalman_predict(last_target, dt)
    
    # Nếu quá lâu (>3s), dừng và xoay tìm
    if time_since_last_detection > 3.0:
        return Twist(vx=0, vy=0, wz=0.2)  # Xoay tìm
```

### 10.3 Target ở sau lưng

```python
if abs(θ_target) > 120°:  # Target ở phía sau
    # Ưu tiên xoay trước, không tiến
    vx = 0
    wz = sign(θ_target) × wz_max
```

### 10.4 Vật cản quá gần (emergency)

```python
min_dist = min(lidar_scan.ranges)
if min_dist < emergency_dist:  # < 0.3m
    # Emergency: dừng tiến, chỉ đi ngang/lùi
    vx = 0
    vy = escape_direction × vy_max
```

### 10.5 Dao động giữa 2 valleys

```python
# Nếu valley thay đổi liên tục giữa 2 giá trị
if oscillation_detected(valley_history):
    # Lock vào valley đầu tiên trong 1s
    best_valley = valley_history[0]
    lock_timer = 1.0
```

---

## 11. Kế hoạch triển khai

### 11.1 Phase 1: VFH Core (2-3 ngày)

```
[ ] Tạo package vfh/
[ ] Implement histogram.py
    [ ] Build polar histogram từ LaserScan
    [ ] Apply robot radius mask
    [ ] Unit tests
[ ] Implement valley_finder.py
    [ ] Find valleys với threshold
    [ ] Score valleys
    [ ] Unit tests
```

### 11.2 Phase 2: APF + Holonomic (2 ngày)

```
[ ] Implement apf_smoother.py
    [ ] Attractive force
    [ ] Repulsive force
    [ ] Blending với VFH direction
[ ] Implement holonomic_decomp.py
    [ ] Decompose θ_final → vx, vy, wz
    [ ] Safety constraints
```

### 11.3 Phase 3: Integration (2 ngày)

```
[ ] Tạo vfh_planner.py (ROS2 node)
    [ ] Subscribe /scan
    [ ] Subscribe target từ person_detector
    [ ] Publish /cmd_vel_vfh
[ ] Sửa person_detector.py
    [ ] Publish /person_detector/target_polar
[ ] Sửa velocity_arbiter.py
    [ ] Subscribe /cmd_vel_vfh thay vì /cmd_vel_person
    [ ] Simplify logic (chỉ safety check)
[ ] Update launch file
```

### 11.4 Phase 4: Testing & Tuning (3-5 ngày)

```
[ ] Test trong simulation (nếu có)
[ ] Test trên robot thật
    [ ] Môi trường trống
    [ ] Có vật cản tĩnh
    [ ] Có vật cản động (người khác)
    [ ] Target di chuyển nhanh
[ ] Tuning parameters
[ ] Viết documentation
```

### 11.5 Timeline tổng

```
Week 1: Phase 1 + 2 (Core algorithms)
Week 2: Phase 3 + 4 (Integration + Testing)
Buffer: 2-3 ngày cho debug

---

## 12. Implementation Plan cho Project hiện tại

### 12.1 Mục tiêu

Thay thế hệ thống tránh vật cản hiện tại (reactive merge trong `velocity_arbiter.py`) bằng thuật toán **VFH+** để:
- Tránh vật cản thông minh hơn (không bị local minima)
- Tích hợp thông tin target từ camera + lidar thành một unified planner
- Tận dụng khả năng đi ngang của robot Mecanum (holonomic)

### 12.2 Lưu ý quan trọng

> [!IMPORTANT]
> **Thay đổi kiến trúc**: VFH+ sẽ trở thành node điều khiển chính, thay thế vai trò của `lidar_processor.py` trong việc phát `/cmd_vel_emergency`.

> [!WARNING]
> **Cần testing kỹ**: Đây là thay đổi core control logic. Khuyến nghị test từng phase riêng biệt trước khi tích hợp hoàn toàn.

### 12.3 Files cần tạo mới

#### [NEW] `vfh/__init__.py`
Export các class chính: `PolarHistogram`, `ValleyFinder`, `APFSmoother`, `HolonomicDecomposer`

#### [NEW] `vfh/histogram.py`
```python
class PolarHistogram:
    def build(scan: LaserScan, n_sectors=72) -> np.ndarray:
        """Tạo histogram từ lidar scan"""
        
    def apply_mask(histogram, robot_radius) -> np.ndarray:
        """Mở rộng obstacles theo robot size"""
```

#### [NEW] `vfh/valley_finder.py`
```python
class ValleyFinder:
    def find_valleys(histogram, threshold_factor=0.5) -> List[Valley]:
        """Tìm các khe hở"""
        
    def score_valley(valley, target_angle, current_heading) -> float:
        """Tính điểm cho mỗi valley"""
        
    def select_best(valleys, target_angle, ...) -> Valley:
        """Chọn valley tối ưu"""
```

#### [NEW] `vfh/apf_smoother.py`
```python
class APFSmoother:
    def attractive_force(target_pos) -> np.ndarray:
        """Lực hút về target"""
        
    def repulsive_force(obstacles, d_influence) -> np.ndarray:
        """Lực đẩy từ obstacles"""
        
    def blend(theta_vfh, theta_apf, beta) -> float:
        """Kết hợp VFH + APF"""
```

#### [NEW] `vfh/holonomic_decomp.py`
```python
class HolonomicDecomposer:
    def decompose(theta_final, target_polar, current_heading) -> Tuple[vx, vy, wz]:
        """Phân tách thành vx, vy, wz"""
        
    def safety_clamp(vx, vy, wz, min_obstacle_dist) -> Tuple[vx, vy, wz]:
        """Giới hạn an toàn"""
```

#### [NEW] `vfh_planner.py`
ROS2 Node chính:

| Type | Topic | Message | Description |
|------|-------|---------|-------------|
| Subscribe | `/scan` | LaserScan | Lidar 360° |
| Subscribe | `/person_detector/target_polar` | Float32MultiArray | (θ, distance) của target |
| Publish | `/cmd_vel_vfh` | Twist | Velocity command tối ưu |
| Publish | `/vfh_planner/debug_image` | Image | Debug visualization |

### 12.4 Files cần sửa đổi

#### [MODIFY] `person_detector.py`
Thêm publisher target polar:
```python
# NEW: Publisher cho VFH+
from std_msgs.msg import Float32MultiArray
self.target_polar_pub = self.create_publisher(
    Float32MultiArray, '/person_detector/target_polar', 10
)

# Trong on_image(), sau khi có target_box:
if self.target_box is not None:
    cx, cy = center_of(self.target_box)
    theta = (cx - W/2) / W * camera_fov  # Convert pixel to angle
    distance = self.last_known_depth or 2.0
    msg = Float32MultiArray(data=[theta, distance])
    self.target_polar_pub.publish(msg)
```

#### [MODIFY] `velocity_arbiter.py`
Thêm subscription cho VFH+:
```python
# NEW: Subscribe VFH+ output
self.create_subscription(Twist, '/cmd_vel_vfh', self._cb_vfh, 10)

# Trong _pick():
has_vfh = self._is_fresh('vfh')
if has_vfh:
    return self.latest['vfh'][0]
```

#### [MODIFY] `setup.py`
```python
# Thêm vfh package
packages=[package_name, f'{package_name}.tracking', f'{package_name}.vfh'],

# Thêm entry point
'vfh_planner = mecanum_control.vfh_planner:main',
```

#### [MODIFY] `mecanum.launch.py`
```python
# NEW: VFH+ Planner Node
vfh_planner = Node(
    package='mecanum_control',
    executable='vfh_planner',
    name='vfh_planner',
    output='screen',
    parameters=[
        {'n_sectors': 72},
        {'range_max': 5.0},
        {'vx_max': 0.4},
        {'vy_max': 0.3},
        {'wz_max': 0.5},
    ],
    condition=IfCondition(LaunchConfiguration('use_lidar')),
)
```

### 12.5 Verification Plan

#### Unit Tests

| Test | File | Coverage |
|------|------|----------|
| Histogram Builder | `test_vfh_histogram.py` | Build histogram, sector assignment, robot radius masking |
| Valley Finder | `test_vfh_valley.py` | Find valleys, score calculation, edge cases |
| APF Smoother | `test_vfh_apf.py` | Attractive/repulsive forces, blending |

```bash
# Run unit tests
python3 -m pytest src/mecanum_control/tests/test_vfh_*.py -v
```

#### Integration Test

```bash
# Terminal 1: Run rosbag với lidar data
ros2 bag play /path/to/lidar_test.bag

# Terminal 2: Run VFH+ node
ros2 run mecanum_control vfh_planner

# Terminal 3: Echo output
ros2 topic echo /cmd_vel_vfh
```

#### Manual Testing

| Test | Scenario | Expected |
|------|----------|----------|
| Test 5 | Môi trường trống | Robot theo người mượt mà |
| Test 6 | Có vật cản tĩnh | Robot đi ngang để tránh |
| Test 7 | Có vật cản động | Robot dừng/tránh, sau đó tiếp tục theo |

### 12.6 Rollback Plan

Nếu VFH+ không hoạt động:

1. **Giữ nguyên `lidar_processor.py`** - không xóa file này
2. **Thêm launch argument** `use_vfh` để switch:
```python
use_vfh = DeclareLaunchArgument('use_vfh', default_value='true')

# Chọn node dựa trên argument
lidar_processor: condition=IfCondition(PythonExpression(['not ', LaunchConfiguration('use_vfh')]))
vfh_planner: condition=IfCondition(LaunchConfiguration('use_vfh'))
```
3. **Rollback command:** 
```bash
ros2 launch mecanum_control mecanum.launch.py use_vfh:=false
```

### 12.7 Timeline Estimate

| Phase | Công việc | Thời gian |
|-------|-----------|-----------|
| 1 | VFH Core Module (4 files) | 2-3 giờ |
| 2 | VFH Planner Node | 1-2 giờ |
| 3 | Integration (sửa 4 files) | 1 giờ |
| 4 | Unit Tests | 1 giờ |
| 5 | Manual Testing + Tuning | User dependent |

**Tổng code mới:** ~500-700 dòng  
**Tổng sửa đổi:** ~50-100 dòng

---

## Tài liệu tham khảo

1. **VFH Original Paper**: Borenstein, J., & Koren, Y. (1991). "The vector field histogram-fast obstacle avoidance for mobile robots"
2. **VFH+ Paper**: Ulrich, I., & Borenstein, J. (1998). "VFH+: Reliable obstacle avoidance for fast mobile robots"
3. **Potential Field**: Khatib, O. (1986). "Real-time obstacle avoidance for manipulators and mobile robots"

---

*Tài liệu được tạo cho project Robot Mecanum theo người - ROS2 Humble*


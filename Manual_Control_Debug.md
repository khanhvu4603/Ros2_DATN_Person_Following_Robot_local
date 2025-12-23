# 🐛 Phân tích lỗi Manual Control không hoạt động

## 📋 Tóm tắt vấn đề

**Hiện tượng:** Khi bật chế độ MANUAL và nhấn các nút điều khiển (Forward, Left, Right, etc.), robot **KHÔNG DI CHUYỂN**.

---

## 🔍 Phân tích luồng dữ liệu

### Luồng hoạt động lý thuyết

```
Frontend (ManualControl.jsx)
    ↓ WebSocket
Backend (main.py) - Nhận lệnh move
    ↓ UDP Port 9998
ROS2 (manual_bridge.py) - Nhận UDP → Publish /cmd_vel_manual
    ↓ ROS2 Topic
velocity_arbiter.py - Nhận /cmd_vel_manual → Publish /cmd_vel_arbiter
    ↓ ROS2 Topic
mecanum_controller.py → stm32_communicator.py → Robot di chuyển
```

---

## ❌ Các điểm lỗi tiềm ẩn

### **Lỗi 1: Logic kiểm tra mode trong `velocity_arbiter.py`**

**File:** `velocity_arbiter.py` - Dòng 113-115

```python
# If MANUAL mode, ignore person
if self.current_mode == "MANUAL":
    has_person = False
```

**Vấn đề:**
- Arbiter chỉ **tắt person tracking** khi ở MANUAL mode
- NHƯNG **KHÔNG ƯU TIÊN** manual command lên trên cùng
- Vẫn có thể bị **EMERGENCY** (lidar) chiếm quyền điều khiển

**Thứ tự ưu tiên hiện tại:**
1. EMERGENCY (lidar) - **Luôn ưu tiên cao nhất**
2. MANUAL
3. PERSON

**Kết quả:** Nếu lidar phát hiện vật cản → EMERGENCY sẽ ghi đè lệnh MANUAL!

---

### **Lỗi 2: Điều kiện `safe_to_move` chặn manual control**

**File:** `velocity_arbiter.py` - Dòng 145-148

```python
# 3) Manual
if has_manual:
    if not self.safe_to_move:  # ← ĐÂY LÀ VẤN ĐỀ!
        return self._zero()
    return self.latest['manual'][0]
```

**Vấn đề:**
- Khi `safe_to_move = False` (lidar phát hiện vật cản)
- Manual control **BỊ CHẶN HOÀN TOÀN** → Trả về velocity = 0
- Người dùng **KHÔNG THỂ** điều khiển robot ngay cả khi muốn

**Nguyên nhân `safe_to_move = False`:**
- Topic `/safe_to_move` được publish bởi `lidar_processor.py`
- Khi có vật cản trong vùng an toàn → `safe_to_move = False`

---

### **Lỗi 3: Lệnh MODE không được gửi đúng cách**

**File:** `backend/main.py` - Dòng 47

```python
if msg_str.startswith("MODE:"):
    mode = msg_str.split(":")[1]
    if mode in ["AUTO", "MANUAL"]:
        self.current_mode = mode
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f"Switched mode to: {mode}")
    return  # ← VẤN ĐỀ: Return sớm!
```

**Vấn đề:**
- Sau khi xử lý MODE, hàm **return ngay**
- Không tiếp tục nhận các gói tin UDP tiếp theo
- **BUG LOGIC:** Nên là `continue` chứ không phải `return`

---

### **Lỗi 4: Frontend không kiểm tra kết nối đúng**

**File:** `ManualControl.jsx` - Dòng 9

```javascript
const canControl = isConnected && isRunning && isManual;
```

**Điều kiện để điều khiển:**
- ✅ `isConnected = true` (WebSocket connected)
- ✅ `isRunning = true` (ROS2 đã start)
- ✅ `isManual = true` (Đã bật chế độ MANUAL)

**Kiểm tra:** Đảm bảo cả 3 điều kiện đều đúng!

---

## 🔧 Giải pháp chi tiết

### **Giải pháp 1: Sửa logic ưu tiên trong `velocity_arbiter.py`**

**Mục tiêu:** Khi ở chế độ MANUAL, **MANUAL phải có ưu tiên cao nhất** (trừ trường hợp khẩn cấp thực sự)

**Sửa hàm `_pick()` trong `velocity_arbiter.py`:**

```python
def _pick(self):
    has_emg    = self._is_fresh('emergency')
    has_manual = self._is_fresh('manual')
    has_person = self._is_fresh('person')

    # If MANUAL mode, prioritize manual control
    if self.current_mode == "MANUAL":
        has_person = False  # Ignore person tracking
        
        # 1) MANUAL has highest priority in MANUAL mode
        if has_manual:
            # Allow manual control even when unsafe (user override)
            return self.latest['manual'][0]
        
        # 2) EMERGENCY only if no manual command
        if has_emg:
            return self.latest['emergency'][0]
        
        # 3) Default stop
        return self._zero()
    
    # AUTO mode logic (existing code)
    # 1) Merge EMERGENCY + PERSON
    if has_emg and has_person and self.merge_when_emergency:
        # ... existing merge logic ...
        
    # 2) EMERGENCY alone
    if has_emg:
        return self.latest['emergency'][0]
    
    # 3) Manual (in AUTO mode, should not happen but keep for safety)
    if has_manual:
        if not self.safe_to_move:
            return self._zero()
        return self.latest['manual'][0]
    
    # 4) Person-follow
    if has_person:
        if not self.safe_to_move and not self.allow_person_when_unsafe:
            return self._zero()
        return self.latest['person'][0]
    
    # 5) Default stop
    return self._zero()
```

**Thay đổi chính:**
- ✅ Khi `MANUAL` mode → Manual command **ưu tiên tuyệt đối**
- ✅ Bỏ qua kiểm tra `safe_to_move` cho manual control
- ✅ Người dùng có toàn quyền điều khiển (user override)

---

### **Giải pháp 2: Sửa bug return trong `manual_bridge.py`**

**File:** `manual_bridge.py` - Dòng 47

**Sửa từ:**
```python
self.get_logger().info(f"Switched mode to: {mode}")
return  # ← SAI!
```

**Thành:**
```python
self.get_logger().info(f"Switched mode to: {mode}")
continue  # ← ĐÚNG: Tiếp tục vòng lặp
```

**Hoặc tốt hơn, tách riêng xử lý MODE:**

```python
def _udp_listener(self):
    while rclpy.ok():
        try:
            data, _ = self.sock.recvfrom(1024)
            msg_str = data.decode('utf-8').strip()
            
            # Check for MODE command
            if msg_str.startswith("MODE:"):
                mode = msg_str.split(":")[1]
                if mode in ["AUTO", "MANUAL"]:
                    self.current_mode = mode
                    msg = String()
                    msg.data = mode
                    self.mode_pub.publish(msg)
                    self.get_logger().info(f"Switched mode to: {mode}")
                continue  # ← Tiếp tục nhận gói tin tiếp theo

            # Parse "vx,vy,wz"
            parts = msg_str.split(',')
            if len(parts) == 3:
                vx = float(parts[0])
                vy = float(parts[1])
                wz = float(parts[2])
                
                twist = Twist()
                twist.linear.x = vx
                twist.linear.y = vy
                twist.angular.z = wz
                
                self.pub.publish(twist)
                
        except Exception as e:
            self.get_logger().error(f"UDP receive error: {e}")
```

---

### **Giải pháp 3: Thêm logging để debug**

**Thêm vào `velocity_arbiter.py` - hàm `_loop()`:**

```python
def _loop(self):
    cmd = self._pick()
    
    # Debug logging
    has_emg = self._is_fresh('emergency')
    has_manual = self._is_fresh('manual')
    has_person = self._is_fresh('person')
    
    if has_manual or has_person or has_emg:
        self.get_logger().info(
            f"Mode={self.current_mode} | "
            f"EMG={has_emg} | MAN={has_manual} | PER={has_person} | "
            f"Safe={self.safe_to_move} | "
            f"CMD: vx={cmd.linear.x:.2f} vy={cmd.linear.y:.2f} wz={cmd.angular.z:.2f}"
        )
    
    # Apply smoothing
    cmd = self._smooth_twist(cmd)
    self.last_cmd = cmd
    self.pub.publish(cmd)
```

---

## 🧪 Cách kiểm tra (Debug Steps)

### **Bước 1: Kiểm tra Frontend gửi lệnh**

Mở **Developer Console** (F12) trong browser:

```javascript
// Kiểm tra WebSocket messages
// Khi nhấn nút Forward, phải thấy:
{
  "type": "command",
  "action": "move",
  "direction": "forward"
}

// Khi nhả nút, phải thấy:
{
  "type": "command",
  "action": "stop_move"
}
```

### **Bước 2: Kiểm tra Backend nhận lệnh**

Xem log của Backend (terminal chạy `python main.py`):

```
INFO:     Received command: {'type': 'command', 'action': 'set_mode', 'mode': 'manual'}
INFO:     Sent mode switch command: MODE:MANUAL
INFO:     Received command: {'type': 'command', 'action': 'move', 'direction': 'forward'}
```

### **Bước 3: Kiểm tra ROS2 nhận UDP**

```bash
# Terminal 1: Chạy ROS2
ros2 launch mecanum_control mecanum.launch.py

# Terminal 2: Monitor topic /cmd_vel_manual
ros2 topic echo /cmd_vel_manual

# Khi nhấn Forward, phải thấy:
linear:
  x: 0.3
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0
```

### **Bước 4: Kiểm tra Arbiter output**

```bash
# Monitor topic /cmd_vel_arbiter
ros2 topic echo /cmd_vel_arbiter

# Phải thấy velocity giống /cmd_vel_manual
```

### **Bước 5: Kiểm tra mode switching**

```bash
# Monitor topic /control_mode
ros2 topic echo /control_mode

# Khi toggle MANUAL, phải thấy:
data: 'MANUAL'

# Khi toggle AUTO, phải thấy:
data: 'AUTO'
```

---

## 📊 Checklist debug

| Bước | Kiểm tra | Lệnh | Kết quả mong đợi |
|------|----------|------|------------------|
| 1 | Frontend gửi lệnh | F12 Console | Thấy message `{action: 'move'}` |
| 2 | Backend nhận lệnh | Log backend | `Received command: move` |
| 3 | Backend gửi UDP | `sudo tcpdump -i lo udp port 9998` | Thấy gói tin UDP |
| 4 | ROS2 nhận UDP | `ros2 topic echo /cmd_vel_manual` | Thấy Twist message |
| 5 | Arbiter xử lý | `ros2 topic echo /cmd_vel_arbiter` | Thấy Twist message |
| 6 | Mode switching | `ros2 topic echo /control_mode` | Thấy "MANUAL" |

---

## 🎯 Kết luận

### Nguyên nhân chính (Most Likely)

1. **`velocity_arbiter.py` chặn manual control** khi `safe_to_move = False`
2. **Logic ưu tiên sai:** EMERGENCY vẫn ghi đè MANUAL
3. **Bug `return` trong `manual_bridge.py`** khiến không nhận lệnh sau khi switch mode

### Giải pháp ưu tiên

1. ✅ **Sửa `velocity_arbiter.py`:** Ưu tiên MANUAL tuyệt đối khi ở MANUAL mode
2. ✅ **Sửa `manual_bridge.py`:** Đổi `return` thành `continue`
3. ✅ **Thêm logging:** Debug để xác định chính xác điểm lỗi

---

**Bạn muốn tôi sửa code luôn không?** Hay cần tôi giải thích thêm phần nào?

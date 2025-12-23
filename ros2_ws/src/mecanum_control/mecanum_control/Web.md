# Hệ Thống Web Control & Video Streaming
## Phân Tích Chi Tiết Frontend, Backend và Giao Tiếp với ROS2

---

## 1. Tổng Quan Kiến Trúc (System Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────────────────┐  │
│   │   FRONTEND  │◄───────►│   BACKEND   │◄───────►│         ROS2            │  │
│   │   (React)   │   WS    │  (FastAPI)  │   UDP   │    (Orange Pi 5)        │  │
│   │   Vercel    │         │   Port 8000 │         │                         │  │
│   └─────────────┘         └──────┬──────┘         └─────────────────────────┘  │
│         ▲                        │                           ▲                  │
│         │                        │                           │                  │
│         │    HTTP MJPEG          │                           │                  │
│         │◄───────────────────────┤                           │                  │
│         │    /video              │        UDP 9999           │                  │
│         │                        │◄──────────────────────────┤                  │
│         │                        │     (JPEG frames)         │                  │
│         │                        │                           │                  │
│         │                        │        UDP 9998           │                  │
│         │                        ├──────────────────────────►│                  │
│         │                        │   (velocity commands)     │                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Thành Phần Chính

| Component | Technology | Port | Chức năng |
|-----------|------------|------|-----------|
| **Frontend** | React + Vite + TailwindCSS | 5173 (dev) | UI điều khiển, hiển thị video |
| **Backend** | FastAPI + Uvicorn | 8000 | WebSocket server, video relay, command bridge |
| **ROS2** | rclpy + OpenCV | - | Person detection, motor control |
| **UDP Video** | Socket | 9999 | Stream JPEG frames từ ROS2 → Backend |
| **UDP Control** | Socket | 9998 | Gửi velocity commands Backend → ROS2 |

---

## 2. Video Streaming Pipeline

### 2.1 Luồng Dữ Liệu Video

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO STREAMING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌─────────┐ │
│  │ Intel D455    │────►│ person_       │────►│   Backend     │────►│ Browser │ │
│  │ RealSense     │     │ detector.py   │     │   (FastAPI)   │     │ <img>   │ │
│  │ Camera        │     │               │     │               │     │         │ │
│  └───────────────┘     └───────┬───────┘     └───────┬───────┘     └─────────┘ │
│                                │                     │                          │
│         ROS2 Topic:            │  UDP Port 9999      │  HTTP GET /video         │
│         /camera/d455/          │  (JPEG, ~70KB)      │  MJPEG Stream             │
│         color/image_raw        │                     │                          │
│                                ▼                     ▼                          │
│                     ┌─────────────────┐   ┌──────────────────┐                  │
│                     │ cv2.imencode()  │   │ generate_frames()│                  │
│                     │ JPEG Quality=70 │   │ multipart/x-mixed│                  │
│                     └─────────────────┘   │ -replace         │                  │
│                                           └──────────────────┘                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Chi Tiết ROS2 → Backend (UDP)

**Sender (person_detector.py):**

```python
# File: person_detector.py, Line 746-753
def publish_debug(self, frame, pboxes, target_box, vmean, depth_m):
    # ...draw debug info on frame...
    
    if self.enable_udp:
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            # Send via UDP to backend
            self.udp_sock.sendto(buffer.tobytes(), (self.udp_host, self.udp_port))
            # UDP_HOST = "127.0.0.1", UDP_PORT = 9999
```

**Receiver (backend/main.py):**

```python
# File: main.py, Line 272-298
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 9999))  # Listen on port 9999

def receive_frames_udp():
    global latest_frame
    while True:
        # Receive UDP packet (max 65KB)
        data, addr = udp_socket.recvfrom(65535)
        
        # Decode JPEG to OpenCV frame
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            latest_frame = frame  # Store for MJPEG streaming

# Run in background thread
threading.Thread(target=receive_frames_udp, daemon=True).start()
```

### 2.3 Backend → Frontend (MJPEG)

**MJPEG Generator (backend/main.py):**

```python
# File: main.py, Line 235-253
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            # Encode current frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = buffer.tobytes()
                # MJPEG boundary format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)  # Wait for frame

@app.get("/video")
async def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**Frontend Display (VideoFeed.jsx):**

```jsx
// File: VideoFeed.jsx, Line 4-18
const VIDEO_URL = 'http://100.114.119.34:8000/video';

export const VideoFeed = ({ isRunning }) => {
    const [hasError, setHasError] = useState(false);

    return (
        <div className="w-full aspect-video bg-black rounded-3xl">
            {isRunning && !hasError ? (
                <img
                    src={VIDEO_URL}     // MJPEG stream
                    alt="Robot Camera Feed"
                    className="w-full h-full object-cover"
                    onError={() => setHasError(true)}
                />
            ) : (
                // Placeholder when offline
            )}
        </div>
    );
};
```

### 2.4 Video Format Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Source Resolution** | 640×480 | Resized trong person_detector |
| **JPEG Quality** | 70% | Cân bằng chất lượng/kích thước |
| **Packet Size** | ~30-70 KB | Tùy thuộc độ phức tạp frame |
| **UDP Port** | 9999 | Local communication |
| **Framerate** | ~10-15 FPS | Phụ thuộc processing speed |
| **Protocol** | MJPEG | Motion JPEG over HTTP |

---

## 3. Control Pipeline (Điều Khiển Robot)

### 3.1 Luồng Điều Khiển

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CONTROL COMMAND PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   USER INPUT                                                                    │
│       │                                                                         │
│       ▼                                                                         │
│   ┌───────────────┐     WebSocket      ┌───────────────┐     UDP 9998          │
│   │ ManualControl │──────────────────►│   Backend     │────────────────────►  │
│   │   (React)     │   {action:"move", │   (FastAPI)   │   "vx,vy,wz"          │
│   │               │    direction:...} │               │   or "MODE:AUTO"      │
│   └───────────────┘                   └───────────────┘                        │
│                                                                                 │
│                                              │                                  │
│                                              ▼                                  │
│   ┌───────────────┐     ROS2 Topic    ┌───────────────┐     ROS2 Topic        │
│   │ velocity_     │◄──────────────────│ manual_bridge │◄────────────────────  │
│   │ arbiter       │  /cmd_vel_manual  │   (rclpy)     │   /control_mode       │
│   └───────┬───────┘                   └───────────────┘                        │
│           │                                                                     │
│           │ /cmd_vel_arbiter                                                    │
│           ▼                                                                     │
│   ┌───────────────┐                                                             │
│   │ stm32_        │──────► Motors                                               │
│   │ communicator  │                                                             │
│   └───────────────┘                                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 WebSocket Protocol

**Connection (useWebSocket.js):**

```javascript
// File: useWebSocket.js
const WS_URL = 'ws://100.114.119.34:8000/ws';

const connect = () => {
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        setIsConnected(true);
        // Send initial connect message
        ws.send(JSON.stringify({ type: 'connect', timestamp: Date.now() }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setLastMessage(data);  // Update UI state
    };
};

const sendCommand = (command) => {
    ws.send(JSON.stringify(command));
};
```

**Message Types:**

| Type | Action | Payload | Description |
|------|--------|---------|-------------|
| `command` | `run` | - | Start ROS2 launch |
| `command` | `stop` | - | Stop ROS2 launch |
| `command` | `move` | `direction: "forward"` | Move robot |
| `command` | `stop_move` | - | Stop movement |
| `command` | `set_mode` | `mode: "auto"/"manual"` | Switch control mode |

**Direction Values:**
- `forward`, `backward`
- `left`, `right`
- `rotate_left`, `rotate_right`

### 3.3 Manual Control Implementation

**Frontend (ManualControl.jsx):**

```jsx
// File: ManualControl.jsx, Line 11-19
const sendMove = (direction) => {
    if (!canControl) return;
    onSendCommand({ type: 'command', action: 'move', direction });
};

const sendStop = () => {
    onSendCommand({ type: 'command', action: 'stop_move' });
};

// Button with press/release events
const getButtonProps = (direction) => ({
    onMouseDown: () => sendMove(direction),
    onMouseUp: sendStop,
    onMouseLeave: sendStop,  // Safety: stop if mouse leaves button
    onTouchStart: (e) => { e.preventDefault(); sendMove(direction); },
    onTouchEnd: (e) => { e.preventDefault(); sendStop(); },
});
```

**Backend Handler (main.py):**

```python
# File: main.py, Line 171-196
SPEED = 0.3         # Linear velocity (m/s)
ROTATE_SPEED = 0.5  # Angular velocity (rad/s)

async def start_manual_move(direction: str):
    global current_velocity
    
    vx, vy, w = 0.0, 0.0, 0.0
    
    if direction == "forward":
        vx = SPEED
    elif direction == "backward":
        vx = -SPEED
    elif direction == "left":
        vy = SPEED
    elif direction == "right":
        vy = -SPEED
    elif direction == "rotate_left":
        w = ROTATE_SPEED
    elif direction == "rotate_right":
        w = -ROTATE_SPEED
    
    current_velocity = {"x": vx, "y": vy, "w": w}
    
    # Start sending UDP at 10Hz
    if manual_control_task is None:
        manual_control_task = asyncio.create_task(send_manual_command_loop())
```

**UDP Sender Loop (main.py):**

```python
# File: main.py, Line 157-169
async def send_manual_command_loop():
    global current_velocity
    while True:
        # Format: "vx,vy,wz"
        msg = f"{current_velocity['x']},{current_velocity['y']},{current_velocity['w']}"
        manual_udp_socket.sendto(msg.encode(), (MANUAL_UDP_IP, MANUAL_UDP_PORT))
        # MANUAL_UDP_PORT = 9998
        await asyncio.sleep(0.1)  # 10Hz
```

### 3.4 ROS2 Manual Bridge (manual_bridge.py)

```python
# File: manual_bridge.py, Line 32-61
def _udp_listener(self):
    while rclpy.ok():
        data, _ = self.sock.recvfrom(1024)
        msg_str = data.decode('utf-8').strip()
        
        # Check for MODE command: "MODE:AUTO" or "MODE:MANUAL"
        if msg_str.startswith("MODE:"):
            mode = msg_str.split(":")[1]
            msg = String()
            msg.data = mode
            self.mode_pub.publish(msg)  # Publish to /control_mode
            return
        
        # Parse velocity: "vx,vy,wz"
        parts = msg_str.split(',')
        if len(parts) == 3:
            twist = Twist()
            twist.linear.x = float(parts[0])
            twist.linear.y = float(parts[1])
            twist.angular.z = float(parts[2])
            
            self.pub.publish(twist)  # Publish to /cmd_vel_manual
```

---

## 4. Mode Switching (Auto/Manual)

### 4.1 State Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODE SWITCHING STATE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │                         AUTO MODE                                    │       │
│   │                                                                     │       │
│   │   velocity_arbiter listens to:                                      │       │
│   │     - /cmd_vel_person   ← person_detector                          │       │
│   │     - /cmd_vel_emergency ← lidar_processor                          │       │
│   │                                                                     │       │
│   │   Ignores: /cmd_vel_manual                                          │       │
│   │                                                                     │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                     │                         ▲                                  │
│   Button Toggle     │ "MODE:MANUAL"           │ "MODE:AUTO"                      │
│   (Frontend)        ▼                         │                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │                        MANUAL MODE                                   │       │
│   │                                                                     │       │
│   │   velocity_arbiter listens to:                                      │       │
│   │     - /cmd_vel_manual   ← manual_bridge                             │       │
│   │     - /cmd_vel_emergency ← lidar_processor                          │       │
│   │                                                                     │       │
│   │   Ignores: /cmd_vel_person                                          │       │
│   │                                                                     │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Velocity Arbiter Logic (velocity_arbiter.py)

```python
# File: velocity_arbiter.py, Line 108-157
def _pick(self):
    has_emg    = self._is_fresh('emergency')
    has_manual = self._is_fresh('manual')
    has_person = self._is_fresh('person')
    
    # If MANUAL mode, ignore person
    if self.current_mode == "MANUAL":
        has_person = False
    
    # Priority 1: EMERGENCY (từ LiDAR)
    if has_emg:
        return self.latest['emergency'][0]
    
    # Priority 2: MANUAL (khi ở MANUAL mode)
    if has_manual:
        return self.latest['manual'][0]
    
    # Priority 3: PERSON (khi ở AUTO mode)
    if has_person:
        return self.latest['person'][0]
    
    # Default: STOP
    return Twist()  # Zero velocity
```

---

## 5. ROS2 Launch & Process Management

### 5.1 Backend Control Flow

**Start ROS2 (main.py):**

```python
# File: main.py, Line 83-118
async def start_ros2():
    global ros2_process
    if ros2_process is None:
        cmd = "ros2 launch mecanum_control mecanum.launch.py"
        
        # Create new process group for clean termination
        ros2_process = subprocess.Popen(
            cmd, 
            shell=True, 
            preexec_fn=os.setsid
        )
        
        await manager.broadcast({
            "type": "response",
            "status": "running",
            "pid": ros2_process.pid
        })
```

**Stop ROS2 (main.py):**

```python
# File: main.py, Line 120-148
async def stop_ros2():
    global ros2_process
    if ros2_process:
        # Kill entire process group with SIGINT
        os.killpg(os.getpgid(ros2_process.pid), signal.SIGINT)
        ros2_process = None
        
        await manager.broadcast({
            "type": "response",
            "status": "stopped"
        })
```

---

## 6. Network Configuration

### 6.1 IP & Port Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NETWORK CONFIGURATION                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Orange Pi 5 Plus (Robot)                                                      │
│   ├── IP: 100.114.119.34 (Tailscale VPN)                                        │
│   │                                                                             │
│   ├── Backend (FastAPI)                                                         │
│   │   ├── HTTP: 0.0.0.0:8000                                                    │
│   │   ├── WebSocket: ws://...:8000/ws                                           │
│   │   └── Video: http://...:8000/video                                          │
│   │                                                                             │
│   ├── UDP Listeners (Backend)                                                   │
│   │   ├── Port 9999: Receive video frames from ROS2                             │
│   │   └── Port 9998: Send velocity commands to ROS2                             │
│   │                                                                             │
│   └── ROS2 Nodes                                                                │
│       ├── manual_bridge: Listen UDP 9998                                        │
│       └── person_detector: Send UDP 9999                                        │
│                                                                                 │
│   Frontend (Vercel)                                                             │
│   └── https://your-app.vercel.app                                               │
│       ├── WebSocket → ws://100.114.119.34:8000/ws                               │
│       └── Video → http://100.114.119.34:8000/video                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 CORS Configuration

```python
# File: main.py, Line 22-28
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)
```

---

## 7. UI Components

### 7.1 Component Hierarchy

```
App.jsx
├── Header.jsx                 # Logo, title
├── StatusBadge.jsx            # Connection status indicator
├── VideoFeed.jsx              # MJPEG video display
├── ControlPanel.jsx           # Connect/Run/Stop buttons
└── ManualControl.jsx          # D-pad + rotation controls
```

### 7.2 Control Panel

| Button | State Required | Action |
|--------|----------------|--------|
| **Connect** | Disconnected | Initialize WebSocket |
| **Run** | Connected + Not Running | Start ROS2 launch |
| **Stop** | Connected + Running | Stop ROS2 launch |

### 7.3 Manual Control D-Pad

```
            ┌───────┐
            │   ▲   │ forward
            │       │
    ┌───────┼───────┼───────┐
    │   ◄   │   ●   │   ►   │ left/right
    │       │       │       │
    └───────┼───────┼───────┘
            │   ▼   │ backward
            │       │
            └───────┘
    
    ┌───────┐       ┌───────┐
    │  ↺    │       │  ↻    │ rotate_left/right
    └───────┘       └───────┘
```

---

## 8. Error Handling & Resilience

### 8.1 Reconnection Logic

```javascript
// Frontend: Auto-reconnect on disconnect
ws.onclose = () => {
    setIsConnected(false);
    // Could add auto-reconnect here
};

ws.onerror = (error) => {
    console.error('WebSocket Error:', error);
    ws.close();  // Clean close triggers onclose
};
```

### 8.2 Stale Data Handling (velocity_arbiter.py)

```python
# File: velocity_arbiter.py, Line 75-79
def _is_fresh(self, key):
    if key not in self.latest:
        return False
    ts = self.latest[key][1]
    # Consider data stale after 500ms
    return (time.time() - ts) * 1000.0 <= self.stale_ms
```

### 8.3 Video Stream Error

```jsx
// Frontend: Handle video stream errors
<img
    src={VIDEO_URL}
    onError={() => setHasError(true)}  // Show placeholder on error
/>
```

---

## 9. Performance Considerations

### 9.1 Latency Sources

| Component | Latency | Notes |
|-----------|---------|-------|
| Camera → ROS2 | ~33ms | 30 FPS camera |
| ROS2 Processing | ~50-100ms | Person detection + tracking |
| UDP Video | <1ms | Local network |
| JPEG Encode | ~5ms | Quality 70% |
| HTTP Transfer | ~10-50ms | Depends on network |
| Browser Render | ~16ms | 60 FPS display |
| **Total** | **~115-200ms** | End-to-end latency |

### 9.2 Bandwidth Usage

| Stream | Size | Rate | Bandwidth |
|--------|------|------|-----------|
| Video (JPEG) | ~50KB | 10 FPS | ~500 KB/s = 4 Mbps |
| Control (UDP) | ~20 bytes | 10 Hz | ~200 B/s |
| WebSocket | ~100 bytes | Event-based | ~1 KB/s |

---

## 10. Security Notes

> **⚠️ Warning**: Cấu hình hiện tại là cho **development only**:

| Risk | Current Setting | Production Fix |
|------|-----------------|----------------|
| CORS | `allow_origins=["*"]` | Whitelist specific domains |
| WebSocket | No auth | Add JWT/session auth |
| UDP | No encryption | Use VPN (Tailscale) ✅ |
| HTTP Video | No auth | Add token-based access |

---

## 11. Summary

### 11.1 Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend Framework** | React 18 + Vite |
| **Styling** | TailwindCSS |
| **State Management** | React Hooks (useState, useEffect) |
| **Backend Framework** | FastAPI + Uvicorn |
| **Real-time Comm** | WebSocket (native) |
| **Video Streaming** | MJPEG over HTTP |
| **Inter-process Comm** | UDP Sockets |
| **Robot Framework** | ROS2 Humble (rclpy) |

### 11.2 Data Flow Summary

```
Frontend (React)
    │
    ├── [WebSocket] ──────────────────────────────► Backend (FastAPI)
    │   Commands: run, stop, move, set_mode              │
    │   Responses: status, mode_changed                  │
    │                                                    │
    └── [HTTP GET /video] ◄───────────────────────── MJPEG Stream
                                                         │
                                    ┌────────────────────┘
                                    │
                            [UDP 9999 IN]  Video frames
                            [UDP 9998 OUT] Velocity commands
                                    │
                                    ▼
                              ROS2 Nodes
                    ├── manual_bridge (control)
                    ├── person_detector (video)
                    ├── velocity_arbiter (priority)
                    └── stm32_communicator (motors)
```

### 11.3 Key Features

1. **Real-time Video Streaming**: MJPEG qua HTTP với latency thấp
2. **Bi-directional Control**: WebSocket cho commands, UDP cho velocity
3. **Mode Switching**: Auto/Manual với priority arbitration
4. **Process Management**: Start/Stop ROS2 từ web
5. **Mobile-friendly**: Touch events cho điều khiển trên điện thoại
6. **VPN Support**: Hoạt động qua Tailscale

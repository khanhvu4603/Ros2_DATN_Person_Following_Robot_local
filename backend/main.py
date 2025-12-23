import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import subprocess
import os
import signal
import logging
import logging
from typing import List
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ros2_process = None
latest_frame = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received command: {data}")
            
            if data.get("type") == "command":
                action = data.get("action")
                if action == "run":
                    await start_ros2()
                elif action == "stop":
                    await stop_ros2()
                elif action == "move":
                    direction = data.get("direction")
                    await start_manual_move(direction)
                elif action == "stop_move":
                    await stop_manual_move()
                elif action == "set_mode":
                    mode = data.get("mode") # "auto" or "manual"
                    await set_control_mode(mode)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def start_ros2():
    global ros2_process
    if ros2_process is None:
        try:
            # Command to run ROS2 launch file
            # Note: Adjust the command to match your environment
            cmd = "ros2 launch mecanum_control mecanum.launch.py"
            
            # Use setsid to create a new process group so we can kill the whole tree later
            ros2_process = subprocess.Popen(
                cmd, 
                shell=True, 
                preexec_fn=os.setsid
                # stdout/stderr inherited by default, so logs will appear in terminal
            )
            
            await manager.broadcast({
                "type": "response",
                "status": "running",
                "pid": ros2_process.pid,
                "message": "ROS2 process started"
            })
            logger.info(f"Started ROS2 process with PID: {ros2_process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start ROS2: {e}")
            await manager.broadcast({
                "type": "error",
                "message": f"Failed to start ROS2: {str(e)}"
            })
    else:
        await manager.broadcast({
            "type": "response",
            "status": "running",
            "message": "ROS2 process is already running"
        })

async def stop_ros2():
    global ros2_process
    if ros2_process:
        try:
            # Kill the process group
            os.killpg(os.getpgid(ros2_process.pid), signal.SIGINT)
            # Wait a bit for graceful shutdown, then force kill if needed
            # For now just SIGINT
            
            ros2_process = None
            await manager.broadcast({
                "type": "response",
                "status": "stopped",
                "message": "ROS2 process stopped"
            })
            logger.info("Stopped ROS2 process")
            
        except Exception as e:
            logger.error(f"Failed to stop ROS2: {e}")
            await manager.broadcast({
                "type": "error",
                "message": f"Failed to stop ROS2: {str(e)}"
            })
    else:
        await manager.broadcast({
            "type": "response",
            "status": "stopped",
            "message": "No ROS2 process running"
        })

# --- Manual Control Logic ---
manual_control_task = None
current_velocity = {"x": 0.0, "y": 0.0, "w": 0.0}
MANUAL_UDP_IP = "127.0.0.1"
MANUAL_UDP_PORT = 9998
manual_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

async def send_manual_command_loop():
    global current_velocity
    while True:
        try:
            # Format: "vx,vy,wz"
            msg = f"{current_velocity['x']},{current_velocity['y']},{current_velocity['w']}"
            manual_udp_socket.sendto(msg.encode(), (MANUAL_UDP_IP, MANUAL_UDP_PORT))
            await asyncio.sleep(0.1) # Send at 10Hz
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error sending manual command: {e}")
            await asyncio.sleep(1.0)

async def start_manual_move(direction: str):
    global manual_control_task, current_velocity
    
    # Define velocities
    SPEED = 0.3
    ROTATE_SPEED = 0.5
    
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
    
    if manual_control_task is None or manual_control_task.done():
        manual_control_task = asyncio.create_task(send_manual_command_loop())

async def stop_manual_move():
    global manual_control_task, current_velocity
    current_velocity = {"x": 0.0, "y": 0.0, "w": 0.0}
    
    # We don't cancel the task immediately to ensure a zero velocity packet is sent,
    # or we can just let it run sending zeros. 
    # Better: Cancel it after a short delay or just send one zero packet and cancel.
    # For simplicity/robustness: Let's keep sending zeros for a bit then stop, 
    # OR just keep the loop running if we expect frequent moves.
    # Let's cancel it to save resources when idle.
    
    # Không cancel loop. Chỉ gửi velocity = 0.0
    current_velocity = {"x": 0.0, "y": 0.0, "w": 0.0}


    # Send a final zero packet to be sure
    manual_udp_socket.sendto(b"0.0,0.0,0.0", (MANUAL_UDP_IP, MANUAL_UDP_PORT))

async def set_control_mode(mode: str):
    if mode not in ["auto", "manual"]:
        return
    
    # Send UDP packet "MODE:AUTO" or "MODE:MANUAL"
    msg = f"MODE:{mode.upper()}"
    try:
        manual_udp_socket.sendto(msg.encode(), (MANUAL_UDP_IP, MANUAL_UDP_PORT))
        logger.info(f"Sent mode switch command: {msg}")
        
        # Broadcast new mode to all clients
        await manager.broadcast({
            "type": "response",
            "status": "mode_changed",
            "mode": mode
        })
    except Exception as e:
        logger.error(f"Error sending mode command: {e}")

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Yield a placeholder or wait
            # For now, just a small sleep to avoid busy loop
            import time
            time.sleep(0.1) 

@app.get("/video")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Endpoint to receive frames from ROS2 (Simple HTTP POST for now, or we can use a socket)
# For simplicity in the first iteration, let's add an endpoint that ROS2 node can POST images to.
# Ideally, we should use shared memory or a raw socket for performance, but HTTP POST is easiest to implement first.
# BUT, the plan mentioned "Publish ROS2 Image topic" or "Shared Memory".
# Let's implement a simple UDP receiver or just a global variable setter if we run in the same process (we don't).
# Let's add a POST endpoint for the ROS2 node to send frames to.
# WARNING: HTTP POST for every frame is slow. 
# Better approach: The ROS2 node sends frames via UDP to a specific port, and this backend listens on that port.
# OR: We use a simple ZMQ socket.
# Let's stick to the plan: "Implement Video Streaming endpoint (MJPEG)".
# I need a way to GET the frames.
# I will add a simple UDP listener in a background thread to receive frames.

import numpy as np
import threading


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 9999))

def receive_frames_udp():
    global latest_frame
    count = 0
    print("UDP Listener started on port 9999")
    while True:
        try:
            # Receive packet (max 65535 bytes)
            data, addr = udp_socket.recvfrom(65535)
            
            if count % 30 == 0:
                print(f"Received UDP packet from {addr}, size: {len(data)}")
            count += 1
            
            # Decode JPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                latest_frame = frame
            else:
                print("Failed to decode frame")
                
        except Exception as e:
            logger.error(f"UDP Receive error: {e}")

# Start UDP listener thread
threading.Thread(target=receive_frames_udp, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

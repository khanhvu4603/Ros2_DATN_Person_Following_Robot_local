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
import threading

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
pending_mode = "AUTO" # Default mode

# ... (ConnectionManager class remains same) ...

async def delayed_set_mode():
    """Wait for ROS2 nodes to spin up, then send the pending mode."""
    logger.info(f"Waiting 5s before sending pending mode: {pending_mode}...")
    await asyncio.sleep(5.0)
    
    # Send UDP packet
    msg = f"MODE:{pending_mode.upper()}"
    try:
        manual_udp_socket.sendto(msg.encode(), (MANUAL_UDP_IP, MANUAL_UDP_PORT))
        logger.info(f"Sent pending mode command: {msg}")
        
        # Broadcast to ensure UI is in sync (though it should already be)
        await manager.broadcast({
            "type": "response",
            "status": "mode_changed",
            "mode": pending_mode
        })
    except Exception as e:
        logger.error(f"Error sending pending mode: {e}")

async def start_ros2():
    global ros2_process
    if ros2_process is None:
        try:
            # Command to run ROS2 launch file
            cmd = "ros2 launch mecanum_control mecanum.launch.py"
            
            # Use setsid to create a new process group
            ros2_process = subprocess.Popen(
                cmd, 
                shell=True, 
                preexec_fn=os.setsid
            )
            
            await manager.broadcast({
                "type": "response",
                "status": "running",
                "pid": ros2_process.pid,
                "message": "ROS2 process started"
            })
            logger.info(f"Started ROS2 process with PID: {ros2_process.pid}")
            
            # Launch background task to apply pending mode
            asyncio.create_task(delayed_set_mode())
            
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

# ... (stop_ros2 remains same) ...

async def set_control_mode(mode: str):
    global pending_mode
    if mode not in ["auto", "manual"]:
        return
    
    # Always update pending_mode
    pending_mode = mode
    
    # If ROS2 is NOT running, just ack the change to UI
    if ros2_process is None:
        logger.info(f"ROS2 not running. Mode '{mode}' queued as pending.")
        await manager.broadcast({
            "type": "response",
            "status": "mode_changed",
            "mode": mode
        })
        return

    # If ROS2 IS running, send immediately
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

# --- WebSocket Video Streaming Logic ---
video_manager = ConnectionManager()

@app.websocket("/ws/video")
async def video_websocket_endpoint(websocket: WebSocket):
    await video_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive control commands or pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        video_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Video WebSocket error: {e}")
        video_manager.disconnect(websocket)

# UDP Listener for Video
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('0.0.0.0', 9999))

def receive_frames_udp():
    print("UDP Video Listener started on port 9999")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            # Receive packet (max 65535 bytes)
            data, addr = udp_socket.recvfrom(65535)
            
            # Broadcast raw JPEG bytes directly to all WebSocket clients
            # This avoids decoding/encoding overhead
            if video_manager.active_connections:
                # We need to run the async broadcast in this sync thread
                # Using run_coroutine_threadsafe is one way, or just fire and forget if possible
                # But ConnectionManager.broadcast is async.
                # Simplest way for this specific loop:
                coro = video_manager.broadcast_bytes(data)
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                
        except Exception as e:
            logger.error(f"UDP Receive error: {e}")

# Add broadcast_bytes to ConnectionManager
async def broadcast_bytes(self, data: bytes):
    for connection in self.active_connections:
        try:
            await connection.send_bytes(data)
        except Exception as e:
            # logger.error(f"Error broadcasting video: {e}")
            pass

# Monkey patch broadcast_bytes into ConnectionManager class (or update class definition)
ConnectionManager.broadcast_bytes = broadcast_bytes

# Start UDP listener thread
threading.Thread(target=receive_frames_udp, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import socket
import cv2
import numpy as np
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending frames to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        # Create a dummy image (random noise or moving rectangle)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a moving rectangle
        t = time.time()
        x = int(320 + 100 * np.sin(t))
        y = int(240 + 100 * np.cos(t))
        cv2.rectangle(frame, (x, y), (x+50, y+50), (0, 255, 0), -1)
        cv2.putText(frame, f"Time: {t:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        if ret:
            # Send via UDP
            sock.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT))
            
        time.sleep(0.033) # ~30 FPS

except KeyboardInterrupt:
    print("Stopped.")

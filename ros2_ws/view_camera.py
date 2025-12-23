import pyrealsense2 as rs
import numpy as np
import cv2

# --------------------- SETUP CAMERA ---------------------
pipeline = rs.pipeline()
config = rs.config()

width, height = 640, 480
fps = 30

config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

print("⏳ Đang khởi động camera...")
pipeline.start(config)

# --------------------- KHỞI TẠO BIẾN ---------------------
rgb_out = None
depth_out = None
recording = False

print("🎥 Đã sẵn sàng. Nhấn ESC để dừng.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert sang numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Xử lý ảnh Depth (Colorize)
        depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # --------------------- SETUP VIDEO WRITER (CHẠY 1 LẦN) ---------------------
        # Chỉ khởi tạo Writer khi đã có frame thực tế để đảm bảo đúng kích thước
        if not recording:
            h, w = color_image.shape[:2]
            
            # LỰA CHỌN CODEC:
            # Option 1: 'avc1' (H.264) -> Tốt cho .mp4 trên Windows
            # Option 2: 'mp4v' -> Cũ, hay lỗi
            # Option 3: 'XVID' -> Tốt cho .avi (Rất ổn định nếu mp4 bị lỗi)
            fourcc = cv2.VideoWriter_fourcc(*"XVID") 
            
            rgb_out = cv2.VideoWriter("rgbV2.mp4", fourcc, fps, (w, h))
            depth_out = cv2.VideoWriter("depthV2.mp4", fourcc, fps, (w, h))

            if not rgb_out.isOpened() or not depth_out.isOpened():
                print("❌ LỖI: Không thể khởi tạo file video. Thử đổi codec sang 'XVID' và đuôi .avi")
                break
            
            print(f"✅ Bắt đầu ghi hình: {w}x{h} @ {fps}fps")
            recording = True

        # --------------------- GHI VIDEO ---------------------
        if recording:
            rgb_out.write(color_image)
            depth_out.write(depth_colored)

        # --------------------- HIỂN THỊ ---------------------
        cv2.imshow("RGB", color_image)
        cv2.imshow("Depth", depth_colored)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            print("⏹ Đang dừng và lưu file...")
            break

finally:
    # Cleanup an toàn
    if rgb_out is not None:
        rgb_out.release()
    if depth_out is not None:
        depth_out.release()
    
    pipeline.stop()
    cv2.destroyAllWindows()
    print("✅ Đã lưu rgb.mp4 và depth.mp4 thành công.")
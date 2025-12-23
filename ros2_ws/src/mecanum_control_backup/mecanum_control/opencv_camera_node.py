#!/usr/bin/env python3
import cv2, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OpenCVCameraNode(Node):
    def __init__(self):
        super().__init__('opencv_camera_node')
        self.declare_parameter('video_device', '/dev/video4')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('show_window', True)

        dev = self.get_parameter('video_device').value
        w   = int(self.get_parameter('width').value)
        h   = int(self.get_parameter('height').value)
        fps = int(self.get_parameter('fps').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.show = bool(self.get_parameter('show_window').value)

        self.pub = self.create_publisher(Image, self.get_parameter('image_topic').value, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        if not self.cap.isOpened():
            self.get_logger().error(f'Không mở được camera: {dev}')
            raise RuntimeError('Camera open failed')

        self.timer = self.create_timer(1.0/max(fps,1), self.tick)
        self.get_logger().info(f'OpenCVCameraNode started on {dev} @ {w}x{h}@{fps}')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Mất khung hình')
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

        if self.show:
            cv2.imshow('OpenCV Camera (ROS2)', frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                rclpy.shutdown()

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if self.show:
                cv2.destroyAllWindows()
        finally:
            super().destroy_node()

def main():
    rclpy.init()
    node = OpenCVCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

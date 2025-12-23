#!/usr/bin/env python3
import cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class OpenCVViewNode(Node):
    def __init__(self):
        super().__init__('opencv_view_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.bridge = CvBridge()
        topic = self.get_parameter('image_topic').value
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f'Viewing topic: {topic} (q/ESC để thoát)')

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('OpenCV Viewer (ROS2)', frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            rclpy.shutdown()

def main():
    rclpy.init()
    node = OpenCVViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

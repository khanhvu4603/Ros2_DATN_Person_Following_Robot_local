#!/usr/bin/env python3
import re
import math
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import threading
import time

LINE_RE = re.compile(
    r'Roll:(?P<roll>-?\d+\.?\d*),\s*Pitch:(?P<pitch>-?\d+\.?\d*),\s*Yaw:(?P<yaw>-?\d+\.?\d*),\s*'
    r'Ax:(?P<ax>-?\d+\.?\d*),\s*Ay:(?P<ay>-?\d+\.?\d*),\s*Az:(?P<az>-?\d+\.?\d*),\s*'
    r'Gx:(?P<gx>-?\d+\.?\d*),\s*Gy:(?P<gy>-?\d+\.?\d*),\s*Gz:(?P<gz>-?\d+\.?\d*)'
)

G = 9.80665

class ImuRawPublisher(Node):
    def __init__(self):
        super().__init__('imu_raw_publisher')

        # params
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('gyro_units', 'rad')
        self.declare_parameter('accel_units', 'ms2')
        self.declare_parameter('log_rate_limit_hz', 1.0)
        self.declare_parameter('read_batch_size', 5)

        port = self.get_parameter('serial_port').value
        baud = int(self.get_parameter('baud_rate').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.gyro_units = str(self.get_parameter('gyro_units').value).lower()
        self.accel_units = str(self.get_parameter('accel_units').value).lower()
        
        # Non-blocking state
        self.serial_buffer = ""
        self._last_warning_log = 0.0
        self._warning_log_interval = 1.0 / float(self.get_parameter('log_rate_limit_hz').value)
        self._error_count = 0
        self._last_error_log = 0.0

        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.01)  # Shorter timeout
            self.get_logger().info(f'[IMU] connected {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(f'[IMU] cannot open serial: {e}')
            raise

        self.pub = self.create_publisher(Imu, '/imu/data_raw', 10)  # Reduced queue size
        self.timer = self.create_timer(0.02, self.read_once)  # 50Hz instead of 100Hz

    def _should_log_error(self):
        current_time = time.time()
        if current_time - self._last_error_log >= 5.0:
            self._last_error_log = current_time
            return True
        return False

    def read_once(self):
        try:
            if not self.ser.in_waiting:
                return
                
            # Batch reading for efficiency
            batch_size = int(self.get_parameter('read_batch_size').value)
            lines = []
            
            for _ in range(batch_size):
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        lines.append(line)
                else:
                    break
            
            # Process batch
            for line in lines:
                self._process_imu_line(line)
                    
        except Exception as e:
            self._error_count += 1
            if self._should_log_error():
                self.get_logger().warn(f'[IMU] read errors: {self._error_count} in 5s')
                self._error_count = 0

    def _process_imu_line(self, line: str):
        m = LINE_RE.search(line.replace(' ', ''))
        if not m:
            return

        ax = float(m['ax']); ay = float(m['ay']); az = float(m['az'])
        gx = float(m['gx']); gy = float(m['gy']); gz = float(m['gz'])

        # convert units if needed
        if self.accel_units == 'g':
            ax *= G; ay *= G; az *= G

        if self.gyro_units == 'deg':
            s = math.pi / 180.0
            gx *= s; gy *= s; gz *= s

        # Quick check with rate limiting
        mag = math.sqrt(ax*ax + ay*ay + az*az)
        current_time = time.time()
        if not (5.0 <= mag <= 15.0):
            if current_time - self._last_warning_log >= self._warning_log_interval:
                self.get_logger().warn(f'[IMU] accel |a| abnormal: {mag:.2f} m/s^2')
                self._last_warning_log = current_time

        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ImuRawPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
import serial, os
from math import sin, cos
import tf_transformations
from tf2_ros import TransformBroadcaster
import threading
import time
import numpy as np


class STM32Communicator(Node):
    def __init__(self):
        super().__init__('stm32_communicator')

        # ======== Parameters ========
        self.declare_parameters(namespace='', parameters=[
            ('serial_port', '/dev/ttyUSB0'),
            ('baud_rate', 115200),
            ('wheel_separation_x', 0.3),
            ('wheel_separation_y', 0.3),
            ('wheel_radius', 0.049),
            ('pwm_max', 130),
            ('pwm_min', 100),                    # NEW: min PWM dynamic
            ('pwm_deadzone', 10),
            ('ema_alpha', 0.4),
            ('odom_publish_rate', 20.0),         # Hz
            ('connection_timeout', 5.0),
            ('serial_write_timeout', 0.01),
            ('control_period', 0.1),              # 10Hz
            ('pwm_filter_alpha', 0.8),            # EMA filter for PWM limit
        ])

        # ======== Internal states ========
        self.serial = None
        self.connected = False
        self.last_cmd_vel = Twist()
        self._ema_pwms = [0, 0, 0, 0]

        # For PWM dynamic scaling
        self.person_distance = 1.5
        self.front_min = 2.0
        self.filtered_pwm_limit = float(self.get_parameter('pwm_max').value)
        self.prev_pwm_limit = self.filtered_pwm_limit

        # Non-blocking serial
        self.serial_lock = threading.Lock()

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.last_time = self.get_clock().now()
        self.odom_publish_interval = 1.0 / float(self.get_parameter('odom_publish_rate').value)

        # ======== Setup ========
        self.setup_ros()
        self.connect_serial()

        # Main control timer
        self.timer = self.create_timer(float(self.get_parameter('control_period').value), self.control_loop)
        self.get_logger().info(f"Control loop running at {1.0/self.get_parameter('control_period').value:.1f} Hz")

    # ------------------------------------------------------------------
    def setup_ros(self):
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_b = TransformBroadcaster(self)

        # Sub thêm khoảng cách người và vật cản
        self.sub_person_dist = self.create_subscription(Float32, '/person_distance', self._cb_person_dist, 10)
        self.sub_front_min = self.create_subscription(Float32, '/lidar_debug/front_min', self._cb_front_min, 10)

    def _cb_person_dist(self, msg: Float32):
        self.person_distance = float(msg.data)

    def _cb_front_min(self, msg: Float32):
        self.front_min = float(msg.data)

    def cmd_cb(self, msg: Twist):
        self.last_cmd_vel = msg

    # ------------------------------------------------------------------
    def connect_serial(self):
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        to = float(self.get_parameter('connection_timeout').value)
        if not os.path.exists(port):
            self.get_logger().error(f"Serial port {port} does not exist!")
            return False
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
            self.serial = serial.Serial(port=port, baudrate=baud, timeout=to)
            self.connected = True
            self.get_logger().info(f"✅ Connected to STM32 at {port}")
            return True
        except Exception as e:
            self.get_logger().error(f"Serial connection error: {e}")
            self.connected = False
            return False

    # ------------------------------------------------------------------
    def compute_dynamic_pwm_limit(self):
        """ PWM max động dựa trên khoảng cách người & vật cản """

        pwm_min = float(self.get_parameter('pwm_min').value)
        pwm_max = float(self.get_parameter('pwm_max').value)
        alpha = float(self.get_parameter('pwm_filter_alpha').value)

        # ---- theo người ----
        d_person = max(0.3, min(2.0, self.person_distance))
        scale_person = (d_person - 0.3) / (2.0 - 0.3)
        pwm_person = pwm_min + (pwm_max - pwm_min) * scale_person

        # ---- theo vật cản ----
        d_front = max(0.2, min(2.0, self.front_min))
        scale_lidar = (d_front - 0.2) / (2.0 - 0.2)
        pwm_lidar = pwm_min + (pwm_max - pwm_min) * scale_lidar

        # ---- chọn cẩn trọng nhất ----
        pwm_limit = min(pwm_person, pwm_lidar)

        # ---- EMA + slope limit ----
        pwm_filtered = alpha * self.filtered_pwm_limit + (1 - alpha) * pwm_limit
        delta = pwm_filtered - self.prev_pwm_limit
        if abs(delta) > 10:
            pwm_filtered = self.prev_pwm_limit + 10 * np.sign(delta)
        self.prev_pwm_limit = pwm_filtered
        self.filtered_pwm_limit = pwm_filtered

        return int(max(pwm_min, min(pwm_max, pwm_filtered)))

    # ------------------------------------------------------------------
    def control_loop(self):
        if not self.connected:
            if not self.connect_serial():
                return
        try:
            self.send_pwm()
            self.publish_odom()
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self.connected = False

    # ------------------------------------------------------------------
    def send_pwm(self):
        wheel_sep_x = float(self.get_parameter('wheel_separation_x').value)
        wheel_sep_y = float(self.get_parameter('wheel_separation_y').value)
        wheel_radius = float(self.get_parameter('wheel_radius').value)
        deadzone = int(self.get_parameter('pwm_deadzone').value)
        alpha = float(self.get_parameter('ema_alpha').value)

        L = wheel_sep_x + wheel_sep_y
        vx = self.last_cmd_vel.linear.x
        vy = self.last_cmd_vel.linear.y
        wz = self.last_cmd_vel.angular.z

        # --- rad/s mỗi bánh ---
        w1 = (vx - vy - L * wz) / wheel_radius
        w2 = (vx + vy + L * wz) / wheel_radius
        w3 = (vx + vy - L * wz) / wheel_radius
        w4 = (vx - vy + L * wz) / wheel_radius

        # --- PWM động ---
        pwm_limit = self.compute_dynamic_pwm_limit()

        raw = [int(w * pwm_limit) for w in [w1, w2, w3, w4]]
        raw = [max(min(p, pwm_limit), -pwm_limit) for p in raw]
        raw = [0 if abs(p) < deadzone else p for p in raw]

        # --- EMA smoothing ---
        smoothed = []
        for i, p in enumerate(raw):
            s = int(alpha * p + (1 - alpha) * self._ema_pwms[i])
            self._ema_pwms[i] = s
            smoothed.append(s)

        packet = f"wheel1:{smoothed[0]},wheel2:{smoothed[1]},wheel3:{smoothed[2]},wheel4:{smoothed[3]}\n"

        # --- Non-blocking serial ---
        def _write_serial():
            try:
                with self.serial_lock:
                    if self.serial and self.serial.is_open:
                        self.serial.write(packet.encode('ascii'))
            except Exception as e:
                self.get_logger().error(f"PWM send error: {e}")
                self.connected = False

        thread = threading.Thread(target=_write_serial)
        thread.daemon = True
        thread.start()

        # Log PWM mỗi 1s cho debug
        now = time.time()
        if not hasattr(self, "_last_log") or now - self._last_log > 1.0:
            self._last_log = now
            # self.get_logger().info(f"PWM_limit={pwm_limit:.0f}, person={self.person_distance:.2f}m, front={self.front_min:.2f}m")

    # ------------------------------------------------------------------
    def publish_odom(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < self.odom_publish_interval:
            return
        self.last_time = now

        vx = self.last_cmd_vel.linear.x
        vy = self.last_cmd_vel.linear.y
        wz = self.last_cmd_vel.angular.z

        self.x += (vx * cos(self.th) - vy * sin(self.th)) * dt
        self.y += (vx * sin(self.th) + vy * cos(self.th)) * dt
        self.th += wz * dt

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        q = tf_transformations.quaternion_from_euler(0, 0, self.th)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_b.sendTransform(t)

    # ------------------------------------------------------------------
    def __del__(self):
        if hasattr(self, 'serial') and self.serial and self.serial.is_open:
            self.serial.close()


def main(args=None):
    rclpy.init(args=args)
    node = STM32Communicator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

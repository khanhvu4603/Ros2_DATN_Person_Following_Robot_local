#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time


class MecanumController(Node):
    """
    Forward /cmd_vel_arbiter -> /cmd_vel.

    Thay đổi tối thiểu:
    - BỎ gate theo /safe_to_move tại controller (để không chặn lệnh emergency từ Arbiter).
    - Cho phép chạy khi LOST/SEARCHING nếu wait_for_detection=False (mặc định).
      Nếu muốn chặn theo trạng thái bám người, set wait_for_detection=True trong launch.
    """

    def __init__(self):
        super().__init__('mecanum_controller')

        # Tham số
        self.wait_for_detection = self.declare_parameter('wait_for_detection', False).value
        self.allowed_states = self.declare_parameter(
            'allowed_states', ['LOCKED', 'SEARCHING', 'LOST']
        ).value
        self.cmd_timeout_s = self.declare_parameter('cmd_timeout_s', 1.0).value

        # Trạng thái
        self.follow_state = 'IDLE'
        self.last_cmd = Twist()
        self.last_cmd_time = 0.0

        # I/O
        self.create_subscription(Twist,  '/cmd_vel_arbiter', self._cb_cmd,   10)
        self.create_subscription(String, '/follow_state',    self._cb_state, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Loop
        self.create_timer(0.05, self._loop)

        self.get_logger().info(
            f'[mecanum_controller] started '
            f'(wait_for_detection={self.wait_for_detection}, '
            f'allowed_states={self.allowed_states}, '
            f'cmd_timeout_s={self.cmd_timeout_s})'
        )

    # ---------- Callbacks ----------
    def _cb_state(self, msg: String):
        self.follow_state = msg.data

    def _cb_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_time = time.time()

    # ---------- Logic ----------
    def _allowed_to_move(self) -> bool:
        # KHÔNG chặn theo safe_to_move ở controller. Arbiter đã quyết định chuyện này.
        if not self.wait_for_detection:
            return True
        return self.follow_state in self.allowed_states

    def _loop(self):
        # Timeout -> dừng
        if (time.time() - self.last_cmd_time) > self.cmd_timeout_s:
            self.pub.publish(Twist())
            return

        # Không được phép theo state -> dừng
        if not self._allowed_to_move():
            self.pub.publish(Twist())
            return

        # Forward lệnh từ arbiter
        self.pub.publish(self.last_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MecanumController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

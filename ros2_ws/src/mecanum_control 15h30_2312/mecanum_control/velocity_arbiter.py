#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

class VelocityArbiter(Node):
    """
    Hợp nhất lệnh vận tốc theo thứ tự ưu tiên, với tuỳ chọn MERGE cải tiến:
      - Khi có cả EMERGENCY (từ lidar) và PERSON (theo người):
          => gộp mượt mà: vx lấy từ PERSON (tiến theo người),
                  vy/wz lấy thành phần lớn hơn giữa EMERGENCY và PERSON,
                  rồi nén biên phẳng để an toàn.
      - Khi chỉ có EMERGENCY: ưu tiên tuyệt đối như cũ.
      - Khi chỉ có MANUAL hoặc PERSON: hành vi như cũ.

    Cải tiến: Thêm cơ chế làm mượt (smoothing) và xử lý tránh vật cản thông minh hơn.
    """

    def __init__(self):
        super().__init__('velocity_arbiter')

        # ---- Params ----
        self.declare_parameter('stale_ms', 500)
        self.declare_parameter('allow_person_when_unsafe', False)
        self.declare_parameter('merge_when_emergency', True)
        self.declare_parameter('max_vx_when_unsafe', 0.25)
        self.declare_parameter('v_planar_cap', 0.6)
        self.declare_parameter('smoothing_factor', 0.5)

        self.stale_ms = self.get_parameter('stale_ms').value
        self.allow_person_when_unsafe = self.get_parameter('allow_person_when_unsafe').value
        self.merge_when_emergency = self.get_parameter('merge_when_emergency').value
        self.max_vx_when_unsafe = float(self.get_parameter('max_vx_when_unsafe').value)
        self.v_planar_cap = float(self.get_parameter('v_planar_cap').value)
        self.smoothing_factor = float(self.get_parameter('smoothing_factor').value)

        # ---- State ----
        self.safe_to_move = True
        self.latest = {}  # key -> (Twist, ts)
        self.last_cmd = Twist()
        self.current_mode = "AUTO"

        # ---- IO ----
        self.create_subscription(Twist, '/cmd_vel_person',    self._cb_person,   10)
        self.create_subscription(Twist, '/cmd_vel_manual',    self._cb_manual,   10)
        self.create_subscription(Twist, '/cmd_vel_emergency', self._cb_emg,      10)
        self.create_subscription(Bool,  '/safe_to_move',      self._cb_safe,     10)
        self.create_subscription(String,'/control_mode',      self._cb_mode,     10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_arbiter', 10)

        # Loop
        self.create_timer(0.05, self._loop)

        self.get_logger().info(
            f'[velocity_arbiter] started (stale_ms={self.stale_ms}, '
            f'allow_person_when_unsafe={self.allow_person_when_unsafe}, '
            f'merge_when_emergency={self.merge_when_emergency}, '
            f'max_vx_when_unsafe={self.max_vx_when_unsafe}, '
            f'v_planar_cap={self.v_planar_cap})'
        )

    # ---------- Callbacks ----------
    def _cb_person(self, msg): self._update('person', msg)
    def _cb_manual(self, msg): self._update('manual', msg)
    def _cb_emg(self, msg):    self._update('emergency', msg)
    def _cb_safe(self, msg):   self.safe_to_move = bool(msg.data)
    def _cb_mode(self, msg):   self.current_mode = msg.data

    # ---------- Helpers ----------
    def _update(self, key, msg):
        self.latest[key] = (msg, time.time())

    def _is_fresh(self, key):
        if key not in self.latest:
            return False
        ts = self.latest[key][1]
        return (time.time() - ts) * 1000.0 <= self.stale_ms

    @staticmethod
    def _zero():
        return Twist()

    @staticmethod
    def _sat(v, lo, hi):
        return max(lo, min(hi, v))

    def _planar_sat(self, vx, vy, vmax):
        n = (vx * vx + vy * vy) ** 0.5
        if n <= 1e-6:
            return 0.0, 0.0
        if n > vmax:
            s = vmax / n
            return vx * s, vy * s
        return vx, vy

    def _smooth_twist(self, new_twist):
        # Apply simple smoothing between last command and new command
        smoothed = Twist()
        alpha = self.smoothing_factor
        smoothed.linear.x = self.last_cmd.linear.x * alpha + new_twist.linear.x * (1 - alpha)
        smoothed.linear.y = self.last_cmd.linear.y * alpha + new_twist.linear.y * (1 - alpha)
        smoothed.angular.z = self.last_cmd.angular.z * alpha + new_twist.angular.z * (1 - alpha)
        return smoothed

    # ---------- Core ----------
    def _pick(self):
        has_emg    = self._is_fresh('emergency')
        has_manual = self._is_fresh('manual')
        has_person = self._is_fresh('person')

        # If MANUAL mode, ignore person
        if self.current_mode == "MANUAL":
            has_person = False

        # 1) Merge EMERGENCY + PERSON => đi xéo
        if has_emg and has_person and self.merge_when_emergency:
            emg = self.latest['emergency'][0]
            per = self.latest['person'][0]
            out = Twist()

            # vx lấy từ PERSON (tiến). Khi unsafe: giới hạn nhỏ và không lùi.
            vx = per.linear.x
            if not self.safe_to_move:
                vx = self._sat(vx, 0.0, self.max_vx_when_unsafe)

            # vy/wz: lấy thành phần có biên độ lớn hơn theo dấu giữa EMG & PERSON
            vy = emg.linear.y if abs(emg.linear.y) >= abs(per.linear.y) else per.linear.y
            wz = emg.angular.z if abs(emg.angular.z) >= abs(per.angular.z) else per.angular.z

            # Nén biên phẳng
            vx, vy = self._planar_sat(vx, vy, self.v_planar_cap)

            out.linear.x  = float(vx)
            out.linear.y  = float(vy)
            out.angular.z = float(wz)
            return out

        # 2) EMERGENCY đơn lẻ -> ưu tiên tuyệt đối
        if has_emg:
            return self.latest['emergency'][0]

        # 3) Manual
        if has_manual:
            if not self.safe_to_move:
                return self._zero()
            return self.latest['manual'][0]

        # 4) Person-follow
        if has_person:
            if not self.safe_to_move and not self.allow_person_when_unsafe:
                return self._zero()
            return self.latest['person'][0]

        # 5) Mặc định dừng
        return self._zero()

    def _loop(self):
        cmd = self._pick()
        # Apply smoothing
        cmd = self._smooth_twist(cmd)
        self.last_cmd = cmd
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import numpy as np
import os
from sensor_msgs.msg import LaserScan

def clamp(x, a, b):
    return a if x < a else b if x > b else x

class ObstacleAvoidance:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
        # Parameters
        self.min_front = config.get('min_safe_distance', 0.60)
        self.min_side  = config.get('safety_zone_sides', 0.50)
        self.fov_front = config.get('angle_range_front', 90.0)
        self.fov_sides = config.get('angle_range_sides', 60.0)
        self.yaw_off_deg = config.get('yaw_offset_deg', 0.0)
        self.emergency_vy = config.get('emergency_vy', 0.22)
        
        self.bypass_min_time_s   = config.get('bypass_min_time_s', 0.40)
        self.bypass_release_hyst = config.get('bypass_release_hysteresis_m', 0.05)
        self.bypass_timeout_s    = config.get('bypass_timeout_s', 3.0)
        
        self.release_on_clear_immediate = config.get('release_on_clear_immediate', True)
        self.clear_debounce_s           = config.get('clear_debounce_s', 0.02)
        self.hold_center_after_rel_s    = config.get('hold_centering_after_release_s', 0.10)
        self.instant_stop_on_clear      = config.get('instant_stop_on_clear', True)
        
        self.gate_mode    = config.get('gate_mode', 'front_only')
        self.vy_slew_rate = config.get('vy_slew_rate', 1.8)
        
        self.enable_center  = config.get('enable_corridor_centering', True)
        self.center_k       = config.get('center_k', 0.6)
        self.center_vy_cap  = config.get('center_vy_cap', 0.20)
        
        self.suppress_close = config.get('suppress_when_target_close', True)
        self.suppress_thr   = config.get('suppress_threshold_m', 0.90)
        
        self.person_ignore_enable       = config.get('person_ignore_enable', True)
        self.person_ignore_margin       = config.get('person_ignore_margin_m', 0.40)
        self.person_ignore_only_locked  = config.get('person_ignore_only_when_locked', True)
        
        self.side_trigger_enable = config.get('side_trigger_enable', True)
        self.side_soft_avoid     = config.get('side_soft_avoid', True)
        self.side_soft_gain      = config.get('side_soft_gain', 0.18)
        
        self.target_distance_m = config.get('target_distance_m', 1.6)
        
        # State
        self.is_locked = False
        self.person_centered = False
        self.person_dist = None
        self.dynamic_front_unsafe = False
        
        self.bypass_active = False
        self.bypass_dir = 0
        self.bypass_start_t = 0.0
        
        self.prev_time = None
        self.prev_vy_cmd = 0.0
        
        self._last_front_clear_t = None
        self._last_release_t = None
        
        self.obstacle_audio_played = False

    def _log_debug(self, msg):
        if self.logger: self.logger.debug(msg)

    def _log_info(self, msg):
        if self.logger: self.logger.info(msg)
        else: print(f"[INFO] {msg}")

    def update_external_state(self, is_locked, person_centered, person_dist, dynamic_front_unsafe):
        self.is_locked = is_locked
        self.person_centered = person_centered
        self.person_dist = person_dist
        self.dynamic_front_unsafe = dynamic_front_unsafe

    def _slew(self, desired_vy: float, tnow: float) -> float:
        if self.prev_time is None:
            self.prev_time = tnow
            self.prev_vy_cmd = desired_vy
            return desired_vy

        dt = max(1e-3, tnow - self.prev_time)
        max_delta = self.vy_slew_rate * dt
        vy = self.prev_vy_cmd

        if desired_vy > vy + max_delta:
            vy += max_delta
        elif desired_vy < vy - max_delta:
            vy -= max_delta
        else:
            vy = desired_vy

        self.prev_time = tnow
        self.prev_vy_cmd = vy
        return vy

    def decide_from_scan(self, scan: LaserScan, tnow: float):
        # Returns: (emergency, vy, wz, unsafe_any, front_unsafe, should_play_sound)
        should_play_sound = False
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)

        rmin = float(scan.range_min) if scan.range_min > 0 else 0.05
        rmax = float(scan.range_max) if scan.range_max > 0 else np.inf
        valid = (ranges >= rmin) & (ranges <= rmax)
        ranges = np.where(valid, ranges, np.inf)

        n = ranges.size
        ang_min = scan.angle_min
        ang_inc = scan.angle_increment
        yaw_off = np.deg2rad(self.yaw_off_deg)

        def angle_to_idx(deg):
            rad = np.deg2rad(deg) + yaw_off
            idx = int(np.round((rad - ang_min) / ang_inc))
            return int(clamp(idx, 0, n - 1))

        # FRONT sector
        half_f = self.fov_front / 2.0
        i0 = angle_to_idx(-half_f)
        i1 = angle_to_idx(+half_f)
        lo_f = min(i0, i1)
        hi_f = max(i0, i1)
        front = ranges[lo_f:hi_f + 1].copy()

        # Mask người theo khoảng cách
        if self.person_ignore_enable and (self.person_dist is not None):
            if (not self.person_ignore_only_locked) or self.is_locked:
                d = float(self.person_dist)
                m = float(self.person_ignore_margin)
                if front.size > 0:
                    mask = np.abs(front - d) <= m
                    front[mask] = np.inf

        # SIDE sectors
        half_s = self.fov_sides / 2.0
        iL0 = angle_to_idx(90 - half_s)
        iL1 = angle_to_idx(90 + half_s)
        left = ranges[min(iL0, iL1):max(iL0, iL1) + 1]

        iR0 = angle_to_idx(-90 - half_s)
        iR1 = angle_to_idx(-90 + half_s)
        right = ranges[min(iR0, iR1):max(iR0, iR1) + 1]

        front_min = float(np.min(front)) if front.size else np.inf
        left_min  = float(np.min(left))  if left.size  else np.inf
        right_min = float(np.min(right)) if right.size else np.inf
        left_avg  = float(np.mean(left)) if left.size  else np.inf
        right_avg = float(np.mean(right)) if right.size else np.inf

        # Kết hợp obstacle tĩnh + động
        front_unsafe = (front_min < self.min_front) or self.dynamic_front_unsafe
        left_unsafe  = (left_min  < self.min_side)
        right_unsafe = (right_min < self.min_side)
        side_unsafe  = (left_unsafe or right_unsafe)
        side_both_unsafe = (left_unsafe and right_unsafe)
        unsafe_any   = bool(front_unsafe or side_unsafe)

        # Track front-clear time
        if not front_unsafe:
            if self._last_front_clear_t is None:
                self._last_front_clear_t = tnow
        else:
            self._last_front_clear_t = None

        # Suppress avoid khi rất gần người
        if self.suppress_close and (self.person_dist is not None) and (self.person_dist <= self.suppress_thr):
            self.bypass_active = False
            self.bypass_dir = 0
            vy_cmd = self._slew(0.0, tnow)
            return False, vy_cmd, 0.0, unsafe_any, front_unsafe, False

        # Quy tắc đặc biệt 2 bên / 3 phía
        if side_both_unsafe and not front_unsafe:
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            return False, 0.0, 0.0, False, False, False

        if side_both_unsafe and front_unsafe:
            self.bypass_active = False
            self.bypass_dir = 0
            self.prev_vy_cmd = 0.0
            self._log_debug('Three-side block (including dynamic front) → stop')
            return False, 0.0, 0.0, True, True, False

        # Front unsafe -> kích bypass
        if front_unsafe and not self.bypass_active:
            if abs(left_min - right_min) < 0.05:
                self.bypass_active = False
                self.bypass_dir = 0
                vy_cmd = self._slew(0.0, tnow)
                return True, vy_cmd, 0.0, unsafe_any, front_unsafe, False
            else:
                self.bypass_dir = +1 if left_min > right_min else -1
                self.bypass_active = True
                self.bypass_start_t = tnow
            
            if self.bypass_active and not self.obstacle_audio_played:
                should_play_sound = True
                self.obstacle_audio_played = True

        # Strict release khi trước thoáng
        if self.bypass_active and self.release_on_clear_immediate and (not front_unsafe):
            if (self._last_front_clear_t is not None) and \
               ((tnow - self._last_front_clear_t) >= self.clear_debounce_s):
                self.bypass_active = False
                self.bypass_dir = 0
                self.prev_vy_cmd = 0.0
                self._last_release_t = tnow
                if self.instant_stop_on_clear and not side_unsafe:
                    return False, 0.0, 0.0, unsafe_any, front_unsafe, should_play_sound

        # Side unsafe & front safe -> né mềm
        trigger_side = (self.side_trigger_enable and side_unsafe and not front_unsafe and not side_both_unsafe)
        if trigger_side and self.side_soft_avoid:
            dir_away = +1 if left_min > right_min else -1
            vy_cmd = self._slew(self.side_soft_gain * float(dir_away), tnow)
            return True, vy_cmd, 0.0, unsafe_any, front_unsafe, should_play_sound

        # Duy trì / nhả bypass
        if self.bypass_active:
            held_enough = (tnow - self.bypass_start_t) >= self.bypass_min_time_s
            clear_enough = (not front_unsafe)
            timeout = (tnow - self.bypass_start_t) >= self.bypass_timeout_s

            if (held_enough and clear_enough) or timeout:
                self.bypass_active = False
                self.bypass_dir = 0
                self.prev_vy_cmd = 0.0
                self._last_release_t = tnow
                self.obstacle_audio_played = False

        if self.bypass_active:
            desired_vy = self.emergency_vy * float(self.bypass_dir)
            vy_cmd = self._slew(desired_vy, tnow)
            return True, vy_cmd, 0.0, unsafe_any, front_unsafe, should_play_sound

        # Vừa nhả: tạm tắt căn giữa
        if self._last_release_t is not None and (tnow - self._last_release_t) < self.hold_center_after_rel_s:
            vy_cmd = 0.0
            return False, vy_cmd, 0.0, unsafe_any, front_unsafe, should_play_sound

        # Corridor centering
        MAX_CORRIDOR = 2.5
        if (not front_unsafe) and (not self.bypass_active) and self.enable_center and \
           (left_min < MAX_CORRIDOR) and (right_min < MAX_CORRIDOR):
            if np.isfinite(left_avg) and np.isfinite(right_avg):
                err_c = (right_avg - left_avg)
                desired_vy = clamp(self.center_k * err_c, -self.center_vy_cap, self.center_vy_cap)
                vy_cmd = self._slew(desired_vy, tnow)
                return False, vy_cmd, 0.0, unsafe_any, front_unsafe, should_play_sound

        # Default: không strafe
        vy_cmd = self._slew(0.0, tnow)
        return False, vy_cmd, 0.0, unsafe_any, front_unsafe, should_play_sound

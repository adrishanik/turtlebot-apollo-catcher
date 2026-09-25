#!/usr/bin/env python3
"""
APOLLO-NAV: Hunter Script (Predictive Interception Controller)
Challenge: TurtleBot Pursuit & Evasion (Tech Zephyr 4.0, IIT Bhubaneswar)
Target Platform: TurtleBot 4 Lite (ROS 2 / Gazebo)
"""

import math
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class ApolloHunterNode(Node):
    def __init__(self):
        super().__init__('hunter_node')

        # TurtleBot 4 Lite physical limits
        self.MAX_LIN_VEL = 0.31        # Max linear velocity (m/s)
        self.MIN_LIN_VEL = 0.05        # Crawl velocity during turns (m/s)
        self.MAX_ANG_VEL = 1.90        # Max angular velocity (rad/s)
        self.ROBOT_RADIUS = 0.17       # Chassis footprint radius (m)
        self.SAFETY_BUFFER = 0.28      # Wall clearance buffer (m)
        self.CAPTURE_DISTANCE = 0.22   # Capture distance threshold (m)

        # Catcher state: [x, y, yaw, speed]
        self.c_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.c_yaw = 0.0
        self.c_speed = 0.0
        self.catcher_ready = False

        # Runner EKF state: [x, y, v, theta, omega]
        self.runner_state = np.zeros(5, dtype=np.float64)
        self.runner_cov = np.eye(5, dtype=np.float64) * 0.1
        self.last_runner_stamp = None
        self.runner_ready = False

        # LiDAR obstacle processing
        self.num_lidar_bins = 72
        self.obstacle_distances = np.full(self.num_lidar_bins, 10.0, dtype=np.float64)
        self.lidar_ready = False

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Actuation publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', reliable_qos)

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self._catcher_odom_cb, sensor_qos)
        self.create_subscription(Odometry, '/runner/odom', self._runner_odom_cb, sensor_qos)
        self.create_subscription(LaserScan, '/scan', self._lidar_cb, sensor_qos)

        # 40 Hz Control Loop
        self.loop_rate_hz = 40.0
        self.dt = 1.0 / self.loop_rate_hz
        self.timer = self.create_timer(self.dt, self._control_cycle)

        self.get_logger().info("Hunter script running at 40 Hz.")

    def _catcher_odom_cb(self, msg: Odometry):
        self.c_pos[0] = msg.pose.pose.position.x
        self.c_pos[1] = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.c_yaw = math.atan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.c_speed = math.hypot(vx, vy)
        self.catcher_ready = True

    def _runner_odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        meas_x = msg.pose.pose.position.x
        meas_y = msg.pose.pose.position.y
        meas_vx = msg.twist.twist.linear.x
        meas_vy = msg.twist.twist.linear.y

        meas_speed = math.hypot(meas_vx, meas_vy)
        meas_theta = math.atan2(meas_vy, meas_vx) if meas_speed > 0.02 else self.runner_state[3]

        if not self.runner_ready:
            self.runner_state = np.array([meas_x, meas_y, meas_speed, meas_theta, 0.0])
            self.last_runner_stamp = stamp
            self.runner_ready = True
            return

        dt = stamp - self.last_runner_stamp
        if dt <= 0.0 or dt > 1.0:
            dt = self.dt
        self.last_runner_stamp = stamp

        self._ekf_predict(dt)
        self._ekf_correct(np.array([meas_x, meas_y, meas_speed, meas_theta]))

    def _lidar_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)

        num_raw = len(ranges)
        if num_raw == 0:
            return

        bin_size = num_raw // self.num_lidar_bins
        for b in range(self.num_lidar_bins):
            chunk = ranges[b * bin_size:(b + 1) * bin_size]
            if len(chunk) > 0:
                self.obstacle_distances[b] = np.min(chunk)

        self.lidar_ready = True

    def _ekf_predict(self, dt: float):
        x, y, v, theta, omega = self.runner_state
        if abs(omega) > 0.001:
            dx = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            dy = (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
        else:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt

        self.runner_state[0] += dx
        self.runner_state[1] += dy
        self.runner_state[3] = self._normalize_angle(theta + omega * dt)

        Q = np.diag([0.05, 0.05, 0.15, 0.08, 0.20]) * dt
        self.runner_cov = self.runner_cov + Q

    def _ekf_correct(self, measurement: np.ndarray):
        H = np.zeros((4, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        R = np.diag([0.02, 0.02, 0.05, 0.08])

        y_res = measurement - H @ self.runner_state
        y_res[3] = self._normalize_angle(y_res[3])

        S = H @ self.runner_cov @ H.T + R
        K = self.runner_cov @ H.T @ np.linalg.inv(S)

        self.runner_state += K @ y_res
        self.runner_state[3] = self._normalize_angle(self.runner_state[3])
        self.runner_cov = (np.eye(5) - K @ H) @ self.runner_cov

    def _solve_apollonius_interception(self) -> np.ndarray:
        rx, ry, rv, r_theta, r_omega = self.runner_state
        cx, cy = self.c_pos

        rv_x = rv * math.cos(r_theta)
        rv_y = rv * math.sin(r_theta)
        dx = rx - cx
        dy = ry - cy

        a = (rv_x**2 + rv_y**2) - (self.MAX_LIN_VEL**2)
        b = 2.0 * (dx * rv_x + dy * rv_y)
        c = dx**2 + dy**2

        t_intercept = None
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                t_linear = -c / b
                if t_linear > 0:
                    t_intercept = t_linear
        else:
            discriminant = b**2 - 4.0 * a * c
            if discriminant >= 0.0:
                t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
                t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
                roots = [t for t in (t1, t2) if t > 0.05]
                if roots:
                    t_intercept = min(roots)

        if t_intercept is None or t_intercept > 12.0:
            return np.array([rx, ry])

        pred_x, pred_y = rx, ry
        sim_step, cur_sim_t, cur_th = 0.1, 0.0, r_theta
        while cur_sim_t < t_intercept:
            pred_x += rv * math.cos(cur_th) * sim_step
            pred_y += rv * math.sin(cur_th) * sim_step
            cur_th += r_omega * sim_step
            cur_sim_t += sim_step

        return np.array([pred_x, pred_y])

    def _filter_heading_with_obstacles(self, desired_heading: float) -> float:
        if not self.lidar_ready:
            return desired_heading

        best_heading = desired_heading
        min_cost = float('inf')
        candidate_offsets = np.linspace(-math.pi / 2, math.pi / 2, 25)

        for offset in candidate_offsets:
            eval_angle = self._normalize_angle(self.c_yaw + offset)
            bin_idx = int(((self._normalize_angle(offset) + math.pi) / (2.0 * math.pi)) * self.num_lidar_bins) % self.num_lidar_bins
            clearance = self.obstacle_distances[bin_idx]

            if clearance < (self.ROBOT_RADIUS + self.SAFETY_BUFFER):
                continue

            angle_diff = abs(self._normalize_angle(eval_angle - desired_heading))
            prox_penalty = 1.0 / max(0.1, clearance)
            total_cost = (1.8 * angle_diff) + (0.6 * prox_penalty)

            if total_cost < min_cost:
                min_cost = total_cost
                best_heading = eval_angle

        return best_heading

    def _control_cycle(self):
        if not self.catcher_ready or not self.runner_ready:
            return

        direct_dist = math.hypot(self.runner_state[0] - self.c_pos[0], self.runner_state[1] - self.c_pos[1])
        if direct_dist <= self.CAPTURE_DISTANCE:
            self._halt_robot()
            return

        target_point = self._solve_apollonius_interception()
        raw_heading = math.atan2(target_point[1] - self.c_pos[1], target_point[0] - self.c_pos[0])
        steering_heading = self._filter_heading_with_obstacles(raw_heading)
        yaw_err = self._normalize_angle(steering_heading - self.c_yaw)

        twist = Twist()
        if abs(yaw_err) > 0.85:
            twist.linear.x = self.MIN_LIN_VEL
            twist.angular.z = float(np.clip(2.5 * yaw_err, -self.MAX_ANG_VEL, self.MAX_ANG_VEL))
        else:
            lin_scale = math.cos(yaw_err)
            speed_candidate = self.MAX_LIN_VEL * max(0.0, lin_scale)
            twist.linear.x = float(np.clip(speed_candidate, self.MIN_LIN_VEL, self.MAX_LIN_VEL))
            twist.angular.z = float(np.clip(1.8 * yaw_err, -self.MAX_ANG_VEL, self.MAX_ANG_VEL))

        self.cmd_pub.publish(twist)

    def _halt_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ApolloHunterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

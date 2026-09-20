#!/usr/bin/env python3
"""
APOLLO-NAV: Apollonius Predictive Pursuit Controller
Event: TurtleBot Pursuit & Evasion Challenge (IIT Bhubaneswar / Tech Zephyr)
Platform: ROS 2 / TurtleBot 4 Lite (Gazebo Sim)

Architecture:
- Dynamic Apollonius Circle Interception (quadratic lead-point solver)
- Extended Kalman Filter (Constant Turn Rate and Velocity / CTRV motion model)
- Vector Field Histogram (VFH) Dynamic Obstacle Avoidance via 2D LiDAR
- Non-Holonomic Kinematics Profiler with adaptive angular velocity dampening
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


class ApolloCatcherNode(Node):
    def __init__(self):
        super().__init__('apollo_catcher_node')

        # -------------------------------------------------------------
        # 1. Hardware & Dynamics Constraints (TurtleBot 4 Lite limits)
        # -------------------------------------------------------------
        self.MAX_LIN_VEL = 0.31        # Maximum linear velocity (m/s)
        self.MIN_LIN_VEL = 0.05        # Minimum crawl speed during tight pivot
        self.MAX_ANG_VEL = 1.90        # Maximum angular rate (rad/s)
        self.ROBOT_RADIUS = 0.17       # Collision footprint radius (m)
        self.SAFETY_BUFFER = 0.28      # Safety expansion around static walls
        self.CAPTURE_DISTANCE = 0.22   # Physical contact / capture threshold (m)

        # -------------------------------------------------------------
        # 2. State Tracking Variables
        # -------------------------------------------------------------
        # Catcher State: [x, y, yaw, v_lin]
        self.c_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.c_yaw = 0.0
        self.c_speed = 0.0
        self.catcher_ready = False

        # Runner EKF State: [x, y, v, theta, omega]
        self.runner_state = np.zeros(5, dtype=np.float64)
        self.runner_cov = np.eye(5, dtype=np.float64) * 0.1
        self.last_runner_stamp = None
        self.runner_ready = False

        # LiDAR Obstacle Map (360-degree sector bins)
        self.num_lidar_bins = 72       # 5-degree angular resolution
        self.obstacle_distances = np.full(self.num_lidar_bins, 10.0, dtype=np.float64)
        self.lidar_ready = False

        # -------------------------------------------------------------
        # 3. ROS 2 Communication Pipelines
        # -------------------------------------------------------------
        # Best-effort sensor QoS profile to eliminate callback pipeline lag
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Standard Reliable QoS for actuators
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Velocity Actuator Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', reliable_qos)

        # Odometry Subscribers (Handles root namespaces and nested team namespaces)
        self.sub_catcher_odom = self.create_subscription(
            Odometry, '/odom', self._catcher_odom_cb, sensor_qos
        )
        self.sub_runner_odom = self.create_subscription(
            Odometry, '/runner/odom', self._runner_odom_cb, sensor_qos
        )

        # 2D Planar LiDAR Subscriber for dynamic avoidance
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._lidar_cb, sensor_qos
        )

        # -------------------------------------------------------------
        # 4. Deterministic 40 Hz Control Loop
        # -------------------------------------------------------------
        self.loop_rate_hz = 40.0
        self.dt = 1.0 / self.loop_rate_hz
        self.timer = self.create_timer(self.dt, self._control_cycle)

        self.get_logger().info("APOLLO-NAV Catcher Engine initialized at 40 Hz.")

    # =================================================================
    # Callback Handlers
    # =================================================================

    def _catcher_odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.c_pos[0] = pos.x
        self.c_pos[1] = pos.y

        # Quaternion to Yaw decomposition
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.c_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Linear speed extraction
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

        # Run EKF Update
        self._ekf_predict(dt)
        self._ekf_correct(np.array([meas_x, meas_y, meas_speed, meas_theta]))

    def _lidar_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)

        # Downsample LiDAR beam array into uniform angular sectors
        num_raw = len(ranges)
        if num_raw == 0:
            return

        bin_size = num_raw // self.num_lidar_bins
        for b in range(self.num_lidar_bins):
            chunk = ranges[b * bin_size:(b + 1) * bin_size]
            if len(chunk) > 0:
                self.obstacle_distances[b] = np.min(chunk)

        self.lidar_ready = True

    # =================================================================
    # EKF State Estimator (CTRV Motion Model)
    # =================================================================

    def _ekf_predict(self, dt: float):
        x, y, v, theta, omega = self.runner_state

        if abs(omega) > 0.001:
            dx = (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
            dy = (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
        else:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt

        dtheta = omega * dt

        # State transition vector
        self.runner_state[0] += dx
        self.runner_state[1] += dy
        self.runner_state[3] = self._normalize_angle(theta + dtheta)

        # Process Noise Covariance
        Q = np.diag([0.05, 0.05, 0.15, 0.08, 0.20]) * dt
        self.runner_cov = self.runner_cov + Q

    def _ekf_correct(self, measurement: np.ndarray):
        H = np.zeros((4, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        R = np.diag([0.02, 0.02, 0.05, 0.08])

        y_res = measurement - H @ self.runner_state
        y_res[3] = self._normalize_angle(y_res[3])

        S = H @ self.runner_cov @ H.T + R
        K = self.runner_cov @ H.T @ np.linalg.inv(S)

        self.runner_state = self.runner_state + K @ y_res
        self.runner_state[3] = self._normalize_angle(self.runner_state[3])
        I = np.eye(5)
        self.runner_cov = (I - K @ H) @ self.runner_cov

    # =================================================================
    # Dynamic Apollonius Lead-Point Solver
    # =================================================================

    def _solve_apollonius_interception(self) -> np.ndarray:
        rx, ry, rv, r_theta, r_omega = self.runner_state
        cx, cy = self.c_pos

        rv_x = rv * math.cos(r_theta)
        rv_y = rv * math.sin(r_theta)

        dx = rx - cx
        dy = ry - cy

        # Solve for root t: ||P_runner(t) - P_catcher(t)|| <= 0
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

        # Fallback to direct pursuit if no future intercept exists within 12 seconds
        if t_intercept is None or t_intercept > 12.0:
            return np.array([rx, ry])

        # Integrate forward along target's predicted turn rate arc
        pred_x = rx
        pred_y = ry
        sim_step = 0.1
        cur_sim_t = 0.0
        cur_th = r_theta

        while cur_sim_t < t_intercept:
            pred_x += rv * math.cos(cur_th) * sim_step
            pred_y += rv * math.sin(cur_th) * sim_step
            cur_th += r_omega * sim_step
            cur_sim_t += sim_step

        return np.array([pred_x, pred_y])

    # =================================================================
    # Vector Field Obstacle Avoidance Filter
    # =================================================================

    def _filter_heading_with_obstacles(self, desired_heading: float) -> float:
        if not self.lidar_ready:
            return desired_heading

        best_heading = desired_heading
        min_cost = float('inf')

        # Sweep candidate forward steering angles from -90 deg to +90 deg
        candidate_offsets = np.linspace(-math.pi / 2, math.pi / 2, 25)

        for offset in candidate_offsets:
            eval_angle = self._normalize_angle(self.c_yaw + offset)
            
            # Map evaluate angle to sector bin
            lidar_angle = self._normalize_angle(offset)
            bin_idx = int(((lidar_angle + math.pi) / (2.0 * math.pi)) * self.num_lidar_bins) % self.num_lidar_bins
            clearance = self.obstacle_distances[bin_idx]

            if clearance < (self.ROBOT_RADIUS + self.SAFETY_BUFFER):
                continue

            # Deviation penalty relative to optimal Apollonius intercept vector
            angle_diff = abs(self._normalize_angle(eval_angle - desired_heading))
            # Proximity penalty
            prox_penalty = 1.0 / max(0.1, clearance)

            total_cost = (1.8 * angle_diff) + (0.6 * prox_penalty)

            if total_cost < min_cost:
                min_cost = total_cost
                best_heading = eval_angle

        return best_heading

    # =================================================================
    # Non-Holonomic Motion Controller Execution
    # =================================================================

    def _control_cycle(self):
        if not self.catcher_ready or not self.runner_ready:
            return

        # 1. Check distance to runner
        direct_dist = math.hypot(self.runner_state[0] - self.c_pos[0], self.runner_state[1] - self.c_pos[1])
        if direct_dist <= self.CAPTURE_DISTANCE:
            self._halt_robot()
            return

        # 2. Derive predictive intercept point
        target_point = self._solve_apollonius_interception()

        # 3. Compute raw desired heading
        raw_heading = math.atan2(target_point[1] - self.c_pos[1], target_point[0] - self.c_pos[0])

        # 4. Reproject through LiDAR obstacle map
        steering_heading = self._filter_heading_with_obstacles(raw_heading)
        yaw_err = self._normalize_angle(steering_heading - self.c_yaw)

        # 5. Non-holonomic velocity generation
        twist = Twist()

        if abs(yaw_err) > 0.85:
            # Pivot in place if angular error is high
            twist.linear.x = self.MIN_LIN_VEL
            twist.angular.z = float(np.clip(2.5 * yaw_err, -self.MAX_ANG_VEL, self.MAX_ANG_VEL))
        else:
            # Forward dash with smooth curve tracking
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
        node = ApolloCatcherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as err:
        sys.stderr.write(f"Runtime execution error in Catcher Node: {err}\n")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

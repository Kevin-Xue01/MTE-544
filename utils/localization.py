import sys
from enum import Enum, auto

import message_filters
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy import init, spin, spin_once
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState

from .ekf import EKF
from .helper import (
    CSVLogger,

    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)
from .ukf import UKF
from .constants import LocalizationMode, PathType
from .config import _config
# kalmanFilter_headers = ["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]

class DifferentialDriveKinematics:
    def __init__(self, wheel_radius, wheel_separation, ticks_per_rev):
        self.r = wheel_radius
        self.L = wheel_separation
        self.ticks_per_rev = ticks_per_rev

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.prev_left_ticks = None
        self.prev_right_ticks = None
        self.prev_time = None

    def update(self, left_ticks, right_ticks, current_time):
        if self.prev_left_ticks is None or self.prev_right_ticks is None:
            self.prev_left_ticks = left_ticks
            self.prev_right_ticks = right_ticks
            self.prev_time = current_time
            return None

        dt = current_time - self.prev_time
        if dt == 0:
            return None

        delta_L = left_ticks - self.prev_left_ticks
        delta_R = right_ticks - self.prev_right_ticks

        theta_L = (delta_L / self.ticks_per_rev) * 2 * np.pi
        theta_R = (delta_R / self.ticks_per_rev) * 2 * np.pi
        d_L = self.r * theta_L
        d_R = self.r * theta_R

        v = (d_L + d_R) / (2 * dt)
        w = (d_R - d_L) / (self.L * dt)

        self.theta += w * dt
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        self.prev_time = current_time

        return v, w, self.x, self.y, self.theta
    
class Localization(Node):
    def __init__(self, type: LocalizationMode = _config.localization_mode, training_iteration: int = _config.training_iteration, path_type: PathType = _config.path_type, dt = 0.1):
        super().__init__("localizer")

        self.pose = np.array([0.0, 0.0, 0.0, self.get_clock().now().to_msg()])
        self.qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=2, history=1, depth=10)
        self.dt = dt

        if type == LocalizationMode.RAW:
            self.initRawSensors()
        elif type == LocalizationMode.EKF:
            self.initKalmanfilter()
        elif type == LocalizationMode.UKF:
            self.initUKF()
        else:
            print("We don't have this type for localization", sys.stderr)
            return
        
        self.last_joint_state = None

        self.est_logger = CSVLogger(f'csv/{training_iteration}/{path_type.name}/{type.name}_robotPose.csv', ["x", "y", "th", "stamp"])
        self.imu_logger = CSVLogger(f'csv/{training_iteration}/{path_type.name}/{type.name}_imu.csv', ["ax", "ay", "stamp"])
        self.noisy_logger = CSVLogger(f"csv/{training_iteration}/{path_type.name}/{type.name}_noisy_odom.csv", ["x", "y", "v", "w", "stamp"])
        self.odom_logger = CSVLogger(f"csv/{training_iteration}/{path_type.name}/{type.name}_odom.csv", ["x", "y", "v", "w", "stamp"])

        # Just for loggining, can remove later
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.log_odom, qos_profile=self.qos)
        self.odom_msg = None
        self.joint_state_msg = None

        #######
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile=self.qos)
        

        # TurtleBot3 Parameters
        self.ticks_per_rev = 4096  # Encoder resolution
        self.kinematics = DifferentialDriveKinematics(
            wheel_radius=0.033,  # TurtleBot3 wheel radius (m)
            wheel_separation=0.16,  # TurtleBot3 track width (m)
            ticks_per_rev=self.ticks_per_rev
        )

        self.current_pose = None

    def joint_state_callback(self, msg: JointState):
        """ Extracts wheel positions (converted from radians to ticks) from /joint_states """
        left_wheel = 'wheel_left_joint'
        right_wheel = 'wheel_right_joint'

        try:
            if left_wheel in msg.name and right_wheel in msg.name:
                left_idx = msg.name.index(left_wheel)
                right_idx = msg.name.index(right_wheel)

                # Convert wheel positions from radians to encoder ticks
                left_ticks = (msg.position[left_idx] * self.ticks_per_rev) / (2 * np.pi)
                right_ticks = (msg.position[right_idx] * self.ticks_per_rev) / (2 * np.pi)
                # print(left_ticks, right_ticks)
                # Get current time
                ros_now = self.get_clock().now()
                current_time = ros_now.nanoseconds * 1e-9

                # Compute kinematics
                result = self.kinematics.update(left_ticks, right_ticks, current_time)
                if result:
                    v, w, x, y, theta = result
                    return v, w, x, y, theta
                return None
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')
        
    def initRawSensors(self):
        # self.create_subscription(odom, "/noisy_odom", self.odom_callback, qos_profile=self.qos)
        self.joint_sub = message_filters.Subscriber(self, JointState, "/joint_states", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.joint_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_raw)
        
    def initKalmanfilter(self):
        x = [0,0,0,0,0,0]

        Q = np.array([
            [5.00000000e-01, 0. , 0. , 0. , 0. , 0. ],
            [0. , 5.00000000e-01, 0. , 0. , 0. , 0. ],
            [0. , 0. , 5.00000000e-01, 0. , 0. , 0. ],
            [0. , 0. , 0. , 1.41957592e+03, 0. , 0. ],
            [0. , 0. , 0. , 0. , 1.03837761e-02, 0. ],
            [0. , 0. , 0. , 0. , 0. , 2.34231199e+01],
        ])

        R = np.array([
            [2.48489894e-05, 0.  , 0.  , 0.  ],
            [0.  , 2.35478560e-05, 0.  , 0.  ],
            [0.  , 0.  , 5.02932847e-03, 0.  ],
            [0.  , 0.  , 0.  , 5.23168700e-05],
        ])

        # Q = np.array([
        #     [0.5, 0. , 0. , 0. , 0. , 0. ],
        #     [0. , 0.5, 0. , 0. , 0. , 0. ],
        #     [0. , 0. , 0.5, 0. , 0. , 0. ],
        #     [0. , 0. , 0. , 0.5, 0. , 0. ],
        #     [0. , 0. , 0. , 0. , 0.5, 0. ],
        #     [0. , 0. , 0. , 0. , 0. , 0.5],
        # ])

        # R = np.array([
        #     [0.25, 0.  , 0.  , 0.  ],
        #     [0.  , 0.25, 0.  , 0.  ],
        #     [0.  , 0.  , 0.25, 0.  ],
        #     [0.  , 0.  , 0.  , 0.25],
        # ])
        
        P = Q.copy()
        
        self.kf = EKF(P,Q,R, x, self.dt)
        
        # self.odom_sub = message_filters.Subscriber(self, odom, "/noisy_odom", qos_profile = self.qos)
        # self.joint_sub = self.create_subscription(
        #     JointState,
        #     '/joint_states',
        #     self.joint_state_callback,
        #     qos_profile=self.qos)
        self.joint_sub = message_filters.Subscriber(self, JointState, "/joint_states", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.joint_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_ekf)
    
    def initUKF(self):
        x = [0,0,0,0,0,0]

        Q = np.array([
            [1.03218359e-01, 0. , 0. , 0. , 0. , 0. ],
            [0. , 1.01333363e-01, 0. , 0. , 0. , 0. ],
            [0. , 0. , 1.00000000e-06, 0. , 0. , 0. ],
            [0. , 0. , 0. , 1.08341521e-01, 0. , 0. ],
            [0. , 0. , 0. , 0. , 1.05329996e-01, 0. ],
            [0. , 0. , 0. , 0. , 0. , 1.01631006e-01],
        ])


        R = np.array([
            [1.08078889e-01, 0.  , 0.  , 0.  ],
            [0.  , 1.00000000e-06, 0.  , 0.  ],
            [0.  , 0.  , 7.82164814e-02, 0.  ],
            [0.  , 0.  , 0.  , 7.96654837e-02],
        ])

        # Q = np.array([
        #     [0.5, 0. , 0. , 0. , 0. , 0. ],
        #     [0. , 0.5, 0. , 0. , 0. , 0. ],
        #     [0. , 0. , 0.5, 0. , 0. , 0. ],
        #     [0. , 0. , 0. , 0.5, 0. , 0. ],
        #     [0. , 0. , 0. , 0. , 0.5, 0. ],
        #     [0. , 0. , 0. , 0. , 0. , 0.5],
        # ])

        # R = np.array([
        #     [0.25, 0.  , 0.  , 0.  ],
        #     [0.  , 0.25, 0.  , 0.  ],
        #     [0.  , 0.  , 0.25, 0.  ],
        #     [0.  , 0.  , 0.  , 0.25],
        # ])

        P = Q.copy()

        self.ukf = UKF(x, P, Q, R, self.dt)

        # self.odom_sub = message_filters.Subscriber(self, odom, "/noisy_odom", qos_profile = self.qos)
        # self.joint_sub = self.create_subscription(
        #     JointState,
        #     '/joint_states',
        #     self.joint_state_callback,
        #     qos_profile=self.qos)
        self.joint_sub = message_filters.Subscriber(self, JointState, "/joint_states", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.joint_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_ukf)
    
    def fusion_callback_raw(self, joint_state_msg: JointState, imu_msg: Imu):
        if joint_state_msg is None:
            return
        result = self.joint_state_callback(joint_state_msg)
        if result is None:
            return
        v, w, x, y, theta = result
        # # TODO Part 3: Use the EKF to perform state estimation
        # # Take the measurements
        # # your measurements are the linear velocity and angular velocity from odom msg
        # # and linear acceleration in x and y from the imu msg
        # # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        # #from odom
        # # v = odom_msg.twist.twist.linear.x
        # # w = odom_msg.twist.twist.angular.z
        # #from IMU
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        # z = np.array([v,w,ax,ay]) #same structure as measurement model
        
        # # Implement the two steps for estimation
        # self.kf.predict()
        # self.kf.update(z)
        
        # # Get the estimate
        # xhat=self.kf.get_states()
        self.pose = np.array([x, y, theta, self.get_clock().now().to_msg()])

        # # TODO Part 4: log your data
        # #presume kf_ax & kf_ay utilize kf values
        # kf_ax = xhat[5]
        # kf_ay = xhat[4]*xhat[3]
        stamp = joint_state_msg.header.stamp.sec + joint_state_msg.header.stamp.nanosec * 1e-9
        
        # stamp = self.pose[3].sec + self.pose[3].nanosec * 1e-9
        self.est_logger.log([x, y, theta, stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([x, y, v, w, stamp])

    def fusion_callback_ekf(self, joint_state_msg: JointState, imu_msg: Imu):
        if joint_state_msg is None:
            return
        result = self.joint_state_callback(joint_state_msg)
        if result is None:
            return
        v, w, x, y, theta = result
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        #from odom
        # v = odom_msg.twist.twist.linear.x
        # w = odom_msg.twist.twist.angular.z
        #from IMU
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        z = np.array([v,w,ax,ay]) #same structure as measurement model
        
        # Implement the two steps for estimation
        self.kf.predict()
        self.kf.update(z)
        
        # Get the estimate
        xhat=self.kf.get_states()
        self.pose=np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

        # # TODO Part 4: log your data
        # #presume kf_ax & kf_ay utilize kf values
        # kf_ax = xhat[5]
        # kf_ay = xhat[4]*xhat[3]

        
        stamp = self.pose[3].sec + self.pose[3].nanosec * 1e-9
        self.est_logger.log([xhat[0], xhat[1], xhat[2], stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([x, y, v, w, stamp])
        self.odom_logger.log([
            self.odom_msg.pose.pose.position.x,
            self.odom_msg.pose.pose.position.y,
            self.odom_msg.twist.twist.linear.x,
            self.odom_msg.twist.twist.angular.z,
            stamp
        ])
    
    def fusion_callback_ukf(self, joint_state_msg: JointState, imu_msg: Imu):
        if joint_state_msg is None:
            return
        result = self.joint_state_callback(joint_state_msg)
        if result is None:
            return
        v, w, x, y, theta = result
        
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        #from odom
        # v = odom_msg.twist.twist.linear.x
        # w = odom_msg.twist.twist.angular.z
        #from IMU
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        z = np.array([v,w,ax, ay]) #same structure as measurement model
        
        # Implement the two steps for estimation
        self.ukf.predict()
        self.ukf.update(z)
        
        # Get the estimate
        xhat=self.ukf.get_states()
        self.pose=np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

        # # TODO Part 4: log your data
        # #presume kf_ax & kf_ay utilize kf values
        # kf_ax = xhat[5]
        # kf_ay = xhat[4]*xhat[3]

        stamp = self.pose[3].sec + self.pose[3].nanosec * 1e-9
        self.est_logger.log([xhat[0], xhat[1], xhat[2], stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([x, y, v, w, stamp])

    def log_odom(self, msg: Odometry):
        self.odom_msg = msg
        
    
    def odom_callback(self, pose_msg):
        self.pose=[pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, euler_from_quaternion(pose_msg.pose.pose.orientation), self.get_clock().now().to_msg()]
        self.est_logger.log(self.pose)

    def getPose(self):
        return self.pose
    
    def joint_states_callback(self, msg):
        self.last_joint_state = msg



if __name__=="__main__":
    
    init()
    
    LOCALIZER=Localization()
    
    spin(LOCALIZER)
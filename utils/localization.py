import sys
from enum import Enum, auto

import message_filters
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as odom
from rclpy import init, spin, spin_once
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState

from .config import _config
from .constants import LocalizationMode
from .ekf import EKF
from .helper import (
    CSVLogger,
    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)
from .ukf import UKF


class Localization(Node):
    def __init__(self, dt = 0.1):
        super().__init__("localizer")

        self.pose = np.array([0.0, 0.0, 0.0, self.get_clock().now().to_msg()])
        self.qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=2, history=1, depth=10)
        self.dt = dt
        self.last_joint_state = None
        self.odom_msg = None

        if _config.localization_mode == LocalizationMode.RAW:
            self.initRawSensors()
        elif _config.localization_mode == LocalizationMode.EKF:
            self.initKalmanfilter()
        elif _config.localization_mode == LocalizationMode.UKF:
            self.initUKF()
        else:
            print("We don't have this type for localization", sys.stderr)
            return
        
        self.est_logger = CSVLogger(f'csv/{_config.localization_mode.name}_robotPose.csv', ["x", "y", "th", "stamp"])
        self.imu_logger = CSVLogger(f'csv/{_config.localization_mode.name}_imu.csv', ["ax", "ay", "stamp"])
        self.noisy_logger = CSVLogger(f"csv/{_config.localization_mode.name}_noisy_odom.csv", ["x", "y", "v", "w", "stamp"])

        self.odom_logger = CSVLogger(f"csv/{_config.localization_mode.name}_odom.csv", ["x", "y", "v", "w", "stamp"])
        self.odom_sub = self.create_subscription(odom, "/odom", self.log_odom, qos_profile=self.qos)
        
    def initRawSensors(self):
        self.create_subscription(odom, "/noisy_odom", self.odom_callback, qos_profile=self.qos)
        
    def initKalmanfilter(self):
        x = [0,0,0,0,0,0]

        Q = np.array([
            [1.00000000e-01, 0. , 0. , 0. , 0. , 0. ],
            [0. , 1.00000000e-01, 0. , 0. , 0. , 0. ],
            [0. , 0. , 1.00000000e-01, 0. , 0. , 0. ],
            [0. , 0. , 0. , 1.60247961e+00, 0. , 0. ],
            [0. , 0. , 0. , 0. , 1.00000000e-06, 0. ],
            [0. , 0. , 0. , 0. , 0. , 2.09662606e-01],
        ])

        R = np.array([
            [1.03422841e-02, 0.  , 0.  , 0.  ],
            [0.  , 1.00000000e-06, 0.  , 0.  ],
            [0.  , 0.  , 5.34596331e-01, 0.  ],
            [0.  , 0.  , 0.  , 3.72916777e-02],
        ])

        P = Q.copy()
        
        self.ekf = EKF(P,Q,R, x, self.dt)
        
        self.odom_sub = message_filters.Subscriber(self, odom, "/noisy_odom", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size = 10, slop = 0.1)
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

        P = Q.copy()

        self.ukf = UKF(x, P, Q, R, self.dt,alpha=1e-3,kappa=0,beta=2)

        self.odom_sub = message_filters.Subscriber(self, odom, "/odom", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_ukf)

    def fusion_callback_ekf(self, odom_msg: odom, imu_msg: Imu):
        if self.odom_msg is None:
            return
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        #from odom
        v = odom_msg.twist.twist.linear.x
        w = odom_msg.twist.twist.angular.z
        #from IMU
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        z = np.array([v,w,ax,ay]) #same structure as measurement model
        
        # Implement the two steps for estimation
        self.ekf.predict()
        self.ekf.update(z)
        
        # Get the estimate
        xhat=self.ekf.get_states()
        self.pose=np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

        # # TODO Part 4: log your data
        # #presume kf_ax & kf_ay utilize kf values
        # kf_ax = xhat[5]
        # kf_ay = xhat[4]*xhat[3]

        stamp = self.pose[3].sec + self.pose[3].nanosec * 1e-9
        self.est_logger.log([xhat[0], xhat[1], xhat[2], stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, stamp])
    
    def fusion_callback_ukf(self, odom_msg: odom, imu_msg: Imu):
        if self.odom_msg is None:
            return
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        #from odom
        v = odom_msg.twist.twist.linear.x
        w = odom_msg.twist.twist.angular.z
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
        self.noisy_logger.log([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, stamp])

    def log_odom(self, msg: odom):
        self.odom_logger.log([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.angular.z, self.get_clock().now()])
        
    
    def odom_callback(self, pose_msg):
        self.pose=[pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, euler_from_quaternion(pose_msg.pose.pose.orientation), self.get_clock().now().to_msg()]
        self.est_logger.log(self.pose)

    def getPose(self):
        return self.pose
    


if __name__=="__main__":
    
    init()
    
    LOCALIZER=Localization()
    
    spin(LOCALIZER)

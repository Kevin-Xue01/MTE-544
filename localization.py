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
from sensor_msgs.msg import Imu

from kalman_filter import kalman_filter
from ukf import ukf
from utilities import (
    CSVLogger,
    LocalizationMode,
    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)

# kalmanFilter_headers = ["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]

class localization(Node):
    def __init__(self, type: LocalizationMode = LocalizationMode.UKF, dt = 0.1):
        super().__init__("localizer")

        self.logger = CSVLogger(f'csv/{type.name}_robot_pose.csv', ["x", "y", "th", "stamp"])
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

    def initRawSensors(self):
        self.create_subscription(odom, "/noisy_odom", self.odom_callback, qos_profile=self.qos)
        
    def initKalmanfilter(self):
        x = [0,0,0,0,0,0]
        
        Q = np.array([
            [0.5, 0. , 0. , 0. , 0. , 0. ],
            [0. , 0.5, 0. , 0. , 0. , 0. ],
            [0. , 0. , 0.5, 0. , 0. , 0. ],
            [0. , 0. , 0. , 0.5, 0. , 0. ],
            [0. , 0. , 0. , 0. , 0.5, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.5],
        ])

        R = np.array([
            [0.25, 0.  , 0.  , 0.  ],
            [0.  , 0.25, 0.  , 0.  ],
            [0.  , 0.  , 0.25, 0.  ],
            [0.  , 0.  , 0.  , 0.25],
        ])
        
        P = Q.copy()
        
        self.kf = kalman_filter(P,Q,R, x, self.dt)
        
        self.odom_sub = message_filters.Subscriber(self, odom, "/noisy_odom", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback)

    def initUKF(self):
        x = [0,0,0,0,0,0]

        Q = np.array([
            [0.01, 0, 0, 0, 0, 0],   # x process noise
            [0, 0.01, 0, 0, 0, 0],   # y process noise
            [0, 0, 0.001, 0, 0, 0],  # theta process noise
            [0, 0, 0, 0.1, 0, 0],    # v process noise
            [0, 0, 0, 0, 0.01, 0],   # w process noise
            [0, 0, 0, 0, 0, 0.001]  # vdot process noise
        ])

        R = np.array([
            [10.0, 0.0, 0.0, 0.0],   # Higher variance for velocity (v), indicating more uncertainty
            [0.0, 0.0001, 0.0, 0.0], # Very small variance for angular velocity (w), very precise
            [0.0, 0.0, 1.0, 0.0],    # Moderate variance for x acceleration (ax)
            [0.0, 0.0, 0.0, 1.0]     # Moderate variance for y acceleration (ay)
        ])

        P = Q.copy()

        self.ukf = ukf(x, P, Q, R, self.dt,alpha=1e-3,kappa=0,beta=2)

        self.odom_sub = message_filters.Subscriber(self, odom, "/odom", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_ukf)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        
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
        self.kf.predict()
        self.kf.update(z)
        
        # Get the estimate
        xhat=self.kf.get_states()
        self.pose=np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

        # # TODO Part 4: log your data
        # #presume kf_ax & kf_ay utilize kf values
        # kf_ax = xhat[5]
        # kf_ay = xhat[4]*xhat[3]

        self.logger.log(self.pose)

    def fusion_callback_ukf(self, odom_msg: odom, imu_msg: Imu):
        
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

        self.logger.log(self.pose)
      
    def odom_callback(self, pose_msg):
        self.pose=[pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, euler_from_quaternion(pose_msg.pose.pose.orientation), self.get_clock().now().to_msg()]
        self.logger.log(self.pose)

    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    spin(LOCALIZER)

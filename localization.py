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

from kalman_filter import kalman_filter
from utilities import (
    CSVLogger,
    LocalizationMode,
    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)

# kalmanFilter_headers = ["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]

class localization(Node):
    def __init__(self, type: LocalizationMode = LocalizationMode.RAW, dt = 0.1):
        super().__init__("localizer")

        self.pose = np.array([0.0, 0.0, 0.0, self.get_clock().now().to_msg()])
        self.qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=2, history=1, depth=10)
        self.dt = dt

        if type == LocalizationMode.RAW:
            self.initRawSensors()
        elif type == LocalizationMode.EKF:
            self.initKalmanfilter()
        else:
            print("We don't have this type for localization", sys.stderr)
            return  
        
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        self.last_joint_state = None

        self.ekf_logger = CSVLogger(f'csv/{type.name}_estimate.csv', ["x", "y", "th", "stamp"])
        self.joint_state_logger = CSVLogger(f'csv/{type.name}_joint_state.csv', ["x", "y", "th", "stamp"])
        self.imu_logger = CSVLogger(f'csv/imu.csv', ["ax", "ay", "stamp"])
        self.noisy_logger = CSVLogger("csv/noisy_odom.csv", ["x", "y", "v", "w", "stamp"])

        # Just for loggining, can remove later
        self.odom_sub = self.create_subscription(odom, "/odom", self.log_odom, qos_profile=self.qos)
        self.odom_logger = CSVLogger("csv/odom.csv", ["x", "y", "v", "w", "stamp"])
        self.odom_msg = None
        
        

    def initRawSensors(self):
        self.create_subscription(odom, "/noisy_odom", self.odom_callback, qos_profile=self.qos)
        
    def initKalmanfilter(self):
        x = [0,0,0,0,0,0]
        
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

        Q = np.array([
            [1.00000000e-01, 0. , 0. , 0. , 0. , 0. ],
            [0. , 1.00000000e-01, 0. , 0. , 0. , 0. ],
            [0. , 0. , 1.00000000e-01, 0. , 0. , 0. ],
            [0. , 0. , 0. , 4.87129354e+01, 0. , 0. ],
            [0. , 0. , 0. , 0. , 5.22819682e+01, 0. ],
            [0. , 0. , 0. , 0. , 0. , 9.59252518e-04],
        ])

        R = np.array([
            [2.48418733e+00, 0.  , 0.  , 0.  ],
            [0.  , 1.00000000e-06, 0.  , 0.  ],
            [0.  , 0.  , 4.10168395e+00, 0.  ],
            [0.  , 0.  , 0.  , 1.73554297e+00],
        ])
        
        P = Q.copy()
        
        self.kf = kalman_filter(P,Q,R, x, self.dt)
        
        self.odom_sub = message_filters.Subscriber(self, odom, "/noisy_odom", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
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
        self.ekf_logger.log([xhat[0], xhat[1], xhat[2], stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, stamp])
        self.odom_logger.log([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, stamp])
      
    def log_odom(self, msg):
        self.odom_msg = msg
    
    def odom_callback(self, pose_msg):
        self.pose=[pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, euler_from_quaternion(pose_msg.pose.pose.orientation), self.get_clock().now().to_msg()]
        self.ekf_logger.log(self.pose)

    def getPose(self):
        return self.pose
    
    def joint_states_callback(self, msg):
        self.last_joint_state = msg



if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    spin(LOCALIZER)

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformListener
import sys

from utilities import Logger

from rclpy.time import Time

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters

rawSensors=0
rawSensors_headers = ["x", "y", "th", "stamp"]
kalmanFilter=1
kalmanFilter_headers = ["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]
odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)

odom_qos = QoSProfile(reliability=2, durability=2, history=1, depth=10)

class OdomVsTFLogger(Node):
    def __init__(self):
        super().__init__('odom_vs_tf_logger')
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.odom_pose = None
        self.tf_pose = None

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        qz, qw = msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        theta = np.arctan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz ** 2))

        self.odom_pose = np.array([x, y, theta])
        self.log_comparison()

    def get_tf_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("odom", "base_footprint", Time())
            x, y = trans.transform.translation.x, trans.transform.translation.y
            qz, qw = trans.transform.rotation.z, trans.transform.rotation.w
            theta = np.arctan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz ** 2))

            self.tf_pose = np.array([x, y, theta])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            self.tf_pose = None

    def log_comparison(self):
        self.get_tf_pose()
        if self.odom_pose is not None and self.tf_pose is not None:
            diff = np.linalg.norm(self.odom_pose[:2] - self.tf_pose[:2])
            print(f"Δ Position: {diff:.4f}, Odom: {self.odom_pose}, TF: {self.tf_pose}")

def main(args=None):
    rclpy.init(args=args)
    node = OdomVsTFLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
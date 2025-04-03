import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from utilities import CSVLogger
from copy import deepcopy

class NoisyOdometry(Node):
    def __init__(self):
        super().__init__("noisy_odometry")

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, "/noisy_odom", 10)

        self.twist_linear_noise_stddev = 0.1
        self.twist_angular_noise_stddev = 0.05

        self.get_logger().info("Noisy Odometry Node Started")

        self.odom_logger = CSVLogger("odom.csv", ["x", "y", "v", "w"])
        self.noisy_logger = CSVLogger("noisy_odom.csv", ["x", "y", "v", "w"])

    def odom_callback(self, msg):
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_v = msg.twist.twist.linear.x
        raw_w = msg.twist.twist.angular.z

        noisy_msg = Odometry()
        noisy_msg.header = msg.header


        noisy_msg.pose.pose = msg.pose.pose
        noisy_twist = deepcopy(msg.twist.twist)
        noisy_twist.linear.x += np.random.normal(0, self.twist_linear_noise_stddev)
        noisy_twist.angular.z += np.random.normal(0, self.twist_angular_noise_stddev)
        noisy_msg.twist.twist = noisy_twist

        self.odom_logger.log([raw_x, raw_y, raw_v, raw_w])
        self.noisy_logger.log([raw_x, raw_y, noisy_twist.linear.x, noisy_twist.angular.z])

        self.odom_pub.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NoisyOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

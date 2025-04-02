import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class NoisyOdometry(Node):
    def __init__(self):
        super().__init__("noisy_odometry")

        # Subscribe to the original odometry topic
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10)

        # Publish the noisy odometry topic
        self.odom_pub = self.create_publisher(Odometry, "/noisy_odom", 10)

        # Noise parameters
        self.position_noise_stddev = 0.01
        self.orientation_noise_stddev = 0.001 
        self.get_logger().info("Noisy Odometry Node Started")

    def odom_callback(self, msg):
        # Extract raw odometry
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_theta = msg.pose.pose.orientation.z  # Approximate yaw

        # Add Gaussian noise
        noisy_x = raw_x + np.random.normal(0, self.position_noise_stddev)
        noisy_y = raw_y + np.random.normal(0, self.position_noise_stddev)
        noisy_theta = raw_theta + np.random.normal(0, self.orientation_noise_stddev)
        # Add Gaussian noise
        # noisy_x = raw_x
        # noisy_y = raw_y
        # noisy_theta = raw_theta

        # Print both values
        print(f"Raw Odom:  x={raw_x:.3f}, y={raw_y:.3f}, theta={raw_theta:.3f}")
        print(f"Noisy Odom: x={noisy_x:.3f}, y={noisy_y:.3f}, theta={noisy_theta:.3f}")
        print("-" * 40)

        # Create a new message with noisy data
        noisy_msg = Odometry()
        noisy_msg.header = msg.header
        noisy_msg.pose.pose.position.x = noisy_x
        noisy_msg.pose.pose.position.y = noisy_y
        noisy_msg.pose.pose.position.z = msg.pose.pose.position.z  # No vertical noise

        noisy_msg.pose.pose.orientation = msg.pose.pose.orientation  # Copy quaternion
        noisy_msg.pose.pose.orientation.z += np.random.normal(0, self.orientation_noise_stddev)
        noisy_msg.pose.pose.orientation.w += np.random.normal(0, self.orientation_noise_stddev)

        noisy_msg.twist = msg.twist  # Copy velocity

        # Publish noisy odometry
        self.odom_pub.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NoisyOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

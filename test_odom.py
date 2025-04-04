import rclpy
import math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class DifferentialDriveDebug(Node):
    def __init__(self):
        super().__init__('diff_drive_debug')

        # Robot parameters (update as needed)
        self.wheel_radius = 0.05  # meters
        self.wheel_separation = 0.3  # meters

        # State variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_rads = None
        self.prev_right_rads = None

        # Subscriptions
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher for computed odometry
        self.odom_pub = self.create_publisher(Odometry, '/custom_odom', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # For tracking Gazebo odometry errors
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_theta = 0.0

    def joint_callback(self, msg):
        # Extract wheel positions
        left_rads = msg.position[0]  # Adjust index if necessary
        right_rads = msg.position[1]  # Adjust index if necessary

        if self.prev_left_rads is None or self.prev_right_rads is None:
            self.prev_left_rads = left_rads
            self.prev_right_rads = right_rads
            return

        # Compute wheel displacements
        delta_left = self.wheel_radius * (left_rads - self.prev_left_rads)
        delta_right = self.wheel_radius * (right_rads - self.prev_right_rads)

        # Compute linear and angular displacement
        delta_s = (delta_left + delta_right) / 2
        delta_theta = (delta_right - delta_left) / self.wheel_separation

        # Update position
        if abs(delta_theta) < 1e-6:
            self.x += delta_s * math.cos(self.theta)
            self.y += delta_s * math.sin(self.theta)
        else:
            R = delta_s / delta_theta
            self.x += R * (math.sin(self.theta + delta_theta) - math.sin(self.theta))
            self.y += -R * (math.cos(self.theta + delta_theta) - math.cos(self.theta))

        self.theta += delta_theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

        # Store new positions
        self.prev_left_rads = left_rads
        self.prev_right_rads = right_rads

        # Publish computed odometry
        self.publish_odom()

    def odom_callback(self, msg):
        # Extract true pose from Gazebo odom
        self.true_x = msg.pose.pose.position.x
        self.true_y = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        _, _, self.true_theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        # Log error
        position_error = math.sqrt((self.x - self.true_x) ** 2 + (self.y - self.true_y) ** 2)
        theta_error = abs(self.theta - self.true_theta)
        self.get_logger().info(f'Position Error: {position_error:.4f}, Theta Error: {theta_error:.4f}')

    def publish_odom(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        # Set orientation
        quat = quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = DifferentialDriveDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

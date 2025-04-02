import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformListener

odom_qos = QoSProfile(reliability=2, durability=2, history=1, depth=10)

class Localization(Node):

    def __init__(self, loggerName="robotPose.csv"):
        super().__init__("localizer")

        self.loc_logger = Logger(loggerName, rawSensors_headers)
        self.pose_odom = np.zeros(3)  # (x, y, theta) from /odom
        self.pose_tf = np.zeros(3)    # (x, y, theta) from /tf

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribing to Odometry
        self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile=odom_qos)

        # Timer for TF lookup
        self.create_timer(0.1, self.get_ground_truth_pose)

    def odom_callback(self, msg):
        """Extract pose from Odometry message."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = np.arctan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz ** 2))

        self.pose_odom = np.array([x, y, theta])
        self.log_comparison()

    def get_ground_truth_pose(self):
        """Extract pose from TF transform."""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform("odom", "base_link", now)

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w
            theta = np.arctan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz ** 2))

            self.pose_tf = np.array([x, y, theta])
            self.log_comparison()

        except Exception as e:
            self.get_logger().warn(f"TF transform unavailable: {e}")

    def log_comparison(self):
        """Log and compare odometry vs ground truth."""
        diff = np.linalg.norm(self.pose_odom[:2] - self.pose_tf[:2])
        self.get_logger().info(f"Odom: {self.pose_odom}, TF: {self.pose_tf}, Error: {diff:.4f}")

        self.loc_logger.log(np.hstack([self.pose_odom, self.pose_tf, diff]))

def main():
    rclpy.init()
    node = Localization()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
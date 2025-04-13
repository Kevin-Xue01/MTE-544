import sys
import message_filters
import numpy as np
from nav_msgs.msg import Odometry
from rclpy import init, spin
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState

from .ekf import EKF
from .helper import CSVLogger
from .ukf import UKF
from .constants import LocalizationMode, PathType
from .config import _config

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
        
        self.est_logger = CSVLogger(f'csv/{training_iteration}/{path_type.name}/{type.name}_robotPose.csv', ["x", "y", "th", "stamp"])
        self.imu_logger = CSVLogger(f'csv/{training_iteration}/{path_type.name}/{type.name}_imu.csv', ["ax", "ay", "stamp"])
        self.noisy_logger = CSVLogger(f"csv/{training_iteration}/{path_type.name}/{type.name}_noisy_odom.csv", ["x", "y", "v", "w", "stamp"])
        self.odom_logger = CSVLogger(f"csv/{training_iteration}/{path_type.name}/{type.name}_odom.csv", ["x", "y", "v", "w", "stamp"])

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.log_odom, qos_profile=self.qos)
        self.odom_msg = None
        self.joint_state_msg = None

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
        left_wheel = 'wheel_left_joint'
        right_wheel = 'wheel_right_joint'

        try:
            if left_wheel in msg.name and right_wheel in msg.name:
                left_idx = msg.name.index(left_wheel)
                right_idx = msg.name.index(right_wheel)

                # Convert wheel positions from radians to encoder ticks
                left_ticks = (msg.position[left_idx] * self.ticks_per_rev) / (2 * np.pi)
                right_ticks = (msg.position[right_idx] * self.ticks_per_rev) / (2 * np.pi)

                ros_now = self.get_clock().now()
                current_time = ros_now.nanoseconds * 1e-9

                result = self.kinematics.update(left_ticks, right_ticks, current_time)
                if result:
                    v, w, x, y, theta = result
                    return v, w, x, y, theta
                return None
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')
        
    def initRawSensors(self):
        self.joint_sub = message_filters.Subscriber(self, JointState, "/joint_states", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.joint_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_raw)
        
    def initKalmanfilter(self):
        x = [0,0,0,0,0,0]

        Q = np.diag([0.5, 0.5, 0.5, 0.80956211, 1.24191514, 0.68991819])
        R = np.diag([1.00000000e-06, 4.85143201e-04, 2.65449949e-04, 1.05829719e-03])


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
        
        self.ekf = EKF(P,Q,R, x, self.dt)
        
        self.joint_sub = message_filters.Subscriber(self, JointState, "/joint_states", qos_profile = self.qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile = self.qos)
        
        time_syncher = message_filters.ApproximateTimeSynchronizer([self.joint_sub, self.imu_sub], queue_size = 10, slop = 0.1)
        time_syncher.registerCallback(self.fusion_callback_ekf)
    
    def initUKF(self):
        x = [0,0,0,0,0,0]

        Q = np.array([
            [1.04807735e-01, 0. , 0. , 0. , 0. , 0. ],
            [0. , 1.00941384e-01, 0. , 0. , 0. , 0. ],
            [0. , 0. , 1.00000000e-06, 0. , 0. , 0. ],
            [0. , 0. , 0. , 1.07750507e-01, 0. , 0. ],
            [0. , 0. , 0. , 0. , 1.04969743e-01, 0. ],
            [0. , 0. , 0. , 0. , 0. , 1.02099372e-01],
        ])

        R = np.array([
            [0.10837514, 0.  , 0.  , 0.  ],
            [0.  , 0.02342533, 0.  , 0.  ],
            [0.  , 0.  , 0.07779667, 0.  ],
            [0.  , 0.  , 0.  , 0.08016707],
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
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        curr_time_msg = self.get_clock().now().to_msg()
        self.pose = np.array([x, y, theta, curr_time_msg])

        stamp = curr_time_msg.sec + curr_time_msg.nanosec * 1e-9
        self.est_logger.log([x, y, theta, stamp])
        self.imu_logger.log([ax, ay, stamp])
        self.noisy_logger.log([x, y, v, w, stamp])
        self.odom_logger.log([
            self.odom_msg.pose.pose.position.x,
            self.odom_msg.pose.pose.position.y,
            self.odom_msg.twist.twist.linear.x,
            self.odom_msg.twist.twist.angular.z,
            stamp
        ])

    def fusion_callback_ekf(self, joint_state_msg: JointState, imu_msg: Imu):
        if joint_state_msg is None:
            return
        result = self.joint_state_callback(joint_state_msg)
        if result is None:
            return
        v, w, x, y, theta = result
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        z = np.array([v,w,ax,ay])
        
        self.ekf.predict()
        self.ekf.update(z)
        
        xhat = self.ekf.get_states()
        self.pose = np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

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
        
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y

        z = np.array([v,w,ax, ay])
        
        self.ukf.predict()
        self.ukf.update(z)
        
        xhat = self.ukf.get_states()
        self.pose = np.array([xhat[0], xhat[1], xhat[2], self.get_clock().now().to_msg()])

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

    def log_odom(self, msg: Odometry):
        self.odom_msg = msg
    
    def getPose(self):
        return self.pose
    
if __name__=="__main__":
    init()
    LOCALIZER=Localization()
    spin(LOCALIZER)
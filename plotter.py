import pandas as pd
from matplotlib import pyplot as plt

ground_truth_file = 'csv/odom.csv' # x,y,v,w # This is ground truth
odom_file = 'csv/noisy_odom.csv' # x,y,v,w
ekf_file = 'csv/EKF_estimate.csv' # x,y,th,stamp
imu_file = 'csv/imu.csv' # x,y,th,stamp

# Read EKF estimate CSV file
ekf_estimate = pd.read_csv(ekf_file)
ground_truth = pd.read_csv(ground_truth_file)

# Plot odometry and EKF estimate trajectories
plt.figure(figsize=(10, 6))
plt.plot(ground_truth['x'], ground_truth['y'], label='Ground Truth', color='blue', alpha=0.7)
plt.plot(ekf_estimate['x'], ekf_estimate['y'], label='EKF Estimate', color='green', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Trajectory: Ground Truth vs EKF Estimate')
plt.legend()
plt.grid()
plt.show()
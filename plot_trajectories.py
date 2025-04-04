import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from utils.config import _config
from utils.constants import LocalizationMode

localization_mode: LocalizationMode = _config.localization_mode
training_iteration: int = _config.training_iteration

base_folder = f'csv/{training_iteration}/'

# Iterate through all subfolders in the base folder
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        # Define file paths
        ground_truth_file = os.path.join(folder_path, f'{localization_mode.name}_odom.csv')  # x,y,v,w
        odom_file = os.path.join(folder_path, f'{localization_mode.name}_noisy_odom.csv')  # x,y,v,w
        ekf_file = os.path.join(folder_path, f'{localization_mode.name}_robotPose.csv')  # x,y,th,stamp

        # Check if all required files exist
        if os.path.exists(ground_truth_file) and os.path.exists(odom_file) and os.path.exists(ekf_file):
            # Read CSV files
            ground_truth = pd.read_csv(ground_truth_file)
            odom = pd.read_csv(odom_file)
            ekf_robotPose = pd.read_csv(ekf_file)

            # Compute MSE for odometry and EKF estimate
            mse_odom = mean_squared_error(ground_truth[['x', 'y']], odom[['x', 'y']])
            mse_ekf = mean_squared_error(ground_truth[['x', 'y']], ekf_robotPose[['x', 'y']])

            # Plot odometry trajectory
            plt.figure(figsize=(10, 6))
            plt.plot(ground_truth['x'], ground_truth['y'], label='Ground Truth', color='black', alpha=0.7, linewidth=2.5)
            plt.plot(odom['x'], odom['y'], label=f'DR (MSE={mse_odom:.4f})', color='orange', alpha=0.7, linestyle='--')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Robot Trajectory - {folder} Path Type')
            plt.legend()
            plt.grid()

            # Save the odometry-only plot to a file
            output_file_odom = os.path.join(folder_path, f'{folder}_{localization_mode.name}_trajectory_odom.png')
            plt.savefig(output_file_odom)
            plt.close()  # Close the plot to avoid displaying it

            # Plot odometry + EKF estimate trajectory
            plt.figure(figsize=(10, 6))
            plt.plot(ground_truth['x'], ground_truth['y'], label='Ground Truth', color='black', alpha=0.7, linewidth=2.5)
            plt.plot(odom['x'], odom['y'], label=f'DR (MSE={mse_odom:.4f})', color='orange', alpha=0.7, linestyle='--')
            plt.plot(ekf_robotPose['x'], ekf_robotPose['y'], label=f'{localization_mode.name} Estimate (MSE={mse_ekf:.4f})', color='green', alpha=0.7, linestyle='-.')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Robot Trajectory - {folder} Path Type')
            plt.legend()
            plt.grid()

            # Save the odometry + EKF estimate plot to a file
            output_file_est = os.path.join(folder_path, f'{folder}_{localization_mode.name}_trajectory_est.png')
            plt.savefig(output_file_est)
            plt.close()  # Close the plot to avoid displaying it
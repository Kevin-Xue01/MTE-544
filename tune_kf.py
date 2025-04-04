import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from utils import EKF
import os


def load_csv(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)

def load_dataset(folder):
    upper_folder = "csv/0/"
    gt_data = load_csv(f'{upper_folder}/{folder}/EKF_odom.csv')
    odom_data = load_csv(f'{upper_folder}/{folder}/EKF_noisy_odom.csv')
    imu_data = load_csv(f'{upper_folder}/{folder}/EKF_imu.csv')
    return odom_data, imu_data, gt_data

def run_kf(q_diag, r_diag, odom_data, imu_data, gt_data):
    Q = np.diag(q_diag)
    R = np.diag(r_diag)

    x0, y0, v0, w0, _ = gt_data[0]
    x_init = np.array([x0, y0, 0.0, w0, v0, 0.0])
    P0 = 0.1 * np.eye(6)
    dt = 0.1
    kf = EKF(P0, Q, R, x_init, dt)
    
    estimates = []
    for i in range(1, len(odom_data)):
        v = odom_data[i, 2]
        w = odom_data[i, 3]
        ax = imu_data[i, 0]
        ay = imu_data[i, 1]
        z = np.array([v, w, ax, ay])
        kf.predict()
        kf.update(z)
        estimates.append(kf.get_states().copy())
    
    return np.array(estimates)

def objective_multi(params, datasets):
    q_diag = params[:6]
    r_diag = params[6:]
    total_mse = 0.0
    for odom_data, imu_data, gt_data in datasets:
        est = run_kf(q_diag, r_diag, odom_data, imu_data, gt_data)
        gt_positions = gt_data[1:, :2]
        kf_positions = est[:, :2]
        mse = np.mean(np.sum((kf_positions - gt_positions)**2, axis=1))
        total_mse += mse
    avg_mse = total_mse / len(datasets)
    return avg_mse

def optimize_noise_params(datasets):
    Q = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    R = [0.25, 0.25, 0.25, 0.25]
    initial_guess = np.concatenate((Q, R))
    bounds = [(1e-6, None)] * 10
    result = minimize(objective_multi, initial_guess, args=(datasets,), bounds=bounds)
    return result

def main():
    folders = ["CIRCLE", "ZIGZAG", "SPORADIC", "SQUARE", "SNAKE", "MIXED"]
    datasets = []
    for folder in folders:
        datasets.append(load_dataset(folder))
    
    result = optimize_noise_params(datasets)
    print("Optimal noise scaling parameters:")
    print("Q diagonals (process noise):", result.x[:6])
    print("R diagonals (measurement noise):", result.x[6:])
    print("Objective (avg. MSE):", result.fun)
    
    base_folder = 'csv/0/'

    # Iterate through all subfolders in the base folder
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            # Define file paths
            ground_truth_file = os.path.join(folder_path, 'EKF_odom.csv')  # x,y,v,w
            odom_file = os.path.join(folder_path, 'EKF_noisy_odom.csv')  # x,y,v,w
            imu_file = os.path.join(folder_path, 'EKF_imu.csv')  # ax,ay
            if os.path.exists(ground_truth_file) and os.path.exists(odom_file) and os.path.exists(imu_file):
                # Load data
                ground_truth = load_csv(ground_truth_file)
                odom = load_csv(odom_file)
                imu = load_csv(imu_file)

                # Use the optimal parameters from tuning
                optimal_q_diag = result.x[:6]
                optimal_r_diag = result.x[6:]
                estimates = run_kf(optimal_q_diag, optimal_r_diag, odom, imu, ground_truth)

                # Plot odometry + EKF estimate trajectory
                plt.figure(figsize=(10, 6))
                plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', color='black', alpha=0.7, linewidth=2.5)
                plt.plot(odom[:, 0], odom[:, 1], label='Odom', color='orange', alpha=0.7, linestyle='--')
                plt.plot(estimates[:, 0], estimates[:, 1], label='EKF Estimate', color='green', alpha=0.7, linestyle='-.')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title(f'Robot Trajectory - {folder} Path Type (Odom + EKF Estimate)')
                plt.legend()
                plt.grid()

                # Save the odometry + EKF estimate plot to a file
                output_file_est = os.path.join(folder_path, f'{folder}_trajectory_trained.png')
                plt.savefig(output_file_est)
                plt.close()  # Close the plot to avoid displaying it

if __name__ == "__main__":
    main()

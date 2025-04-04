import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from ukf import ukf  # Ensure your ukf class is in a file named ukf.py or adjust the import accordingly

def load_csv(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)

def load_dataset(folder):
    upper_folder = "csv/baseline"
    gt_data = load_csv(f'{upper_folder}/{folder}/odom.csv')
    odom_data = load_csv(f'{upper_folder}/{folder}/noisy_odom.csv')
    imu_data = load_csv(f'{upper_folder}/{folder}/imu.csv')
    return odom_data, imu_data, gt_data

def run_ukf(q_diag, r_diag, odom_data, imu_data, gt_data):
    # Build Q and R as diagonal matrices from the provided parameters.
    Q = np.diag(q_diag)  # 6x6 process noise covariance
    R = np.diag(r_diag)  # 4x4 measurement noise covariance

    # Initialize state from the first row of the ground truth data.
    # Adjust indices as necessary depending on your data format.
    x0, y0, v0, w0, _ = gt_data[0]
    x_init = np.array([x0, y0, 0.0, w0, v0, 0.0])
    P0 = 0.1 * np.eye(6)
    dt = 0.1  # Fixed time step (update if timestamps are available)
    
    # Instantiate the UKF filter with the given initial state and noise matrices.
    filter_instance = ukf(x_init, P0, Q, R, dt)
    
    estimates = []
    # Loop through the data and update the filter.
    for i in range(1, len(odom_data)):
        # Build the measurement vector z = [v, w, ax, ay].
        v = odom_data[i, 2]
        w = odom_data[i, 3]
        ax = imu_data[i, 0]
        ay = imu_data[i, 1]
        z = np.array([v, w, ax, ay])
        
        filter_instance.predict()
        filter_instance.update(z)
        
        # Save a copy of the current state estimate.
        estimates.append(filter_instance.x.copy())
        
    return np.array(estimates)

def objective_multi(params, datasets):
    q_diag = params[:6]
    r_diag = params[6:]
    total_mse = 0.0
    for odom_data, imu_data, gt_data in datasets:
        est = run_ukf(q_diag, r_diag, odom_data, imu_data, gt_data)
        gt_positions = gt_data[1:, :2]
        ukf_positions = est[:, :2]
        mse = np.mean(np.sum((ukf_positions - gt_positions)**2, axis=1))
        total_mse += mse
    avg_mse = total_mse / len(datasets)
    return avg_mse

def optimize_noise_params(datasets):
    initial_guess = [0.1] * 10
    bounds = [(1e-6, None)] * 10
    result = minimize(objective_multi, initial_guess, args=(datasets,), bounds=bounds)
    return result

def main():
    folders = ["circular", "zigzag", "snake", "square","sporadic"]
    #, "zigzag", "snake", "square"
    datasets = [load_dataset(folder) for folder in folders]
    
    result = optimize_noise_params(datasets)
    print("Optimal noise scaling parameters:")
    print("Q diagonals (process noise):", result.x[:6])
    print("R diagonals (measurement noise):", result.x[6:])
    print("Objective (avg. MSE):", result.fun)
    
    for idx, (odom_data, imu_data, gt_data) in enumerate(datasets):
        est = run_ukf(result.x[:6], result.x[6:], odom_data, imu_data, gt_data)
        gt_positions = gt_data[1:, :2]
        ukf_positions = est[:, :2]
        
        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground Truth", marker='o', linestyle='-')
        plt.plot(ukf_positions[:, 0], ukf_positions[:, 1], label="UKF Estimate", marker='x', linestyle='--')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Dataset {idx+1}: UKF Fusion - Estimated Trajectory vs. Ground Truth")
        plt.legend()
        plt.show()
if __name__ == "__main__":
    main()

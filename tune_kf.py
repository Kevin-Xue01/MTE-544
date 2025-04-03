import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kalman_filter import kalman_filter

def load_csv(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)

ground_truth_file = 'csv/odom.csv'
odom_file = 'csv/noisy_odom.csv'
imu_file = 'csv/imu.csv'

gt_data = load_csv(ground_truth_file)
odom_data = load_csv(odom_file)
imu_data = load_csv(imu_file)

def run_kf(q_diag, r_diag, odom_data, imu_data, gt_data):
    Q = np.diag(q_diag)
    R = np.diag(r_diag)

    x0, y0, v0, w0, _ = gt_data[0]
    x_init = np.array([x0, y0, 0.0, w0, v0, 0.0])
    P0 = 0.1 * np.eye(6)
    dt = 0.1
    kf = kalman_filter(P0, Q, R, x_init, dt)
    
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

def objective(params, odom_data, imu_data, gt_data):
    q_diag = params[:6]
    r_diag = params[6:]
    est = run_kf(q_diag, r_diag, odom_data, imu_data, gt_data)
    gt_positions = gt_data[1:, :2]
    kf_positions = est[:, :2]
    mse = np.mean(np.sum((kf_positions - gt_positions)**2, axis=1))
    return mse

def optimize_noise_params(odom_data, imu_data, gt_data):
    initial_guess = [1.00000000e-01, 1.00000000e-01, 1.00000000e-01, 4.87129269e+01,
                     5.22819884e+01, 1.00000000e-06, 2.37939464e+00, 1.00000000e-06,
                     4.10168885e+00, 1.86933007e+00]
    bounds = [(1e-6, None)] * 10
    result = minimize(objective, initial_guess, args=(odom_data, imu_data, gt_data), bounds=bounds)
    return result

def main():
    result = optimize_noise_params(odom_data, imu_data, gt_data)
    print("Optimal noise scaling parameters:")
    print("Q diagonals (process noise):", result.x[:6])
    print("R diagonals (measurement noise):", result.x[6:])
    print("Objective MSE:", result.fun)
    
    est = run_kf(result.x[:6], result.x[6:], odom_data, imu_data, gt_data)
    gt_positions = gt_data[1:, :2]
    kf_positions = est[:, :2]
    
    plt.figure()
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground Truth", marker='o', linestyle='-')
    plt.plot(kf_positions[:, 0], kf_positions[:, 1], label="KF Estimate", marker='x', linestyle='--')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("KF Fusion: Estimated Trajectory vs. Ground Truth")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

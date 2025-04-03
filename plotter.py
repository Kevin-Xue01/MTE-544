import pandas as pd
import matplotlib.pyplot as plt

# File paths
noisy_odom_file = 'csv/noisy_odom.csv'
odom_file = 'csv/odom.csv'

# Read CSV files
noisy_odom = pd.read_csv(noisy_odom_file)
odom = pd.read_csv(odom_file)

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot noisy odometry (x, y)
axs[0].scatter(noisy_odom['x'], noisy_odom['y'], label='Noisy Odometry', alpha=0.7, color='red', s=10)
axs[0].scatter(odom['x'], odom['y'], label='Odometry', alpha=0.7, color='blue', s=10)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Odometry vs Noisy Odometry')
axs[0].legend()
axs[0].grid()

# Plot linear velocity (v)
axs[1].plot(noisy_odom['v'], label='Noisy Odometry v', color='red', alpha=0.7)
axs[1].plot(odom['v'], label='Odometry v', color='blue', alpha=0.7)
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Linear Velocity (v)')
axs[1].set_title('Linear Velocity (v) Comparison')
axs[1].legend()
axs[1].grid()

# Plot angular velocity (w)
axs[2].plot(noisy_odom['w'], label='Noisy Odometry w', color='red', alpha=0.7)
axs[2].plot(odom['w'], label='Odometry w', color='blue', alpha=0.7)
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Angular Velocity (w)')
axs[2].set_title('Angular Velocity (w) Comparison')
axs[2].legend()
axs[2].grid()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
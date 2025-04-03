import numpy as np

class ukf:

    def __init__(self, x, P, Q, R, dt, alpha=1e-3, kappa=0, beta=2):
        self.n = len(x) # state dimension
        self.n_sig = 1 + self.n * 2 # number of sigma points

        self.x = np.array(x, dtype=float) # Initial state
        self.P = np.array(P, dtype=float)  # Covariance matrix

        self.Q = np.array(Q, dtype=float) # process noise
        self.R = np.array(R, dtype=float) # measurement noise
        self.dt = dt  # Time step

        # Compute lambda and weights for sigma points
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lmbda)
        
        self.weights_mean = np.full(2 * self.n + 1, 0.5 / (self.n + self.lmbda)) # initialize
        self.weights_cov = np.copy(self.weights_mean) # initialize

        self.weights_mean[0] = self.lmbda / (self.n + self.lmbda)
        self.weights_cov[0] = (self.lmbda / (self.n + self.lmbda)) + (1 - pow(alpha, 2) + beta)

    def generate_sigma_points(self):
        sqrt_P = np.linalg.cholesky(self.P + 1e-6 * np.eye(self.n)) # cholesky decomp with regulatization term
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.x
        for i in range(self.n):
            sigma_points[i + 1] = self.x + self.gamma * sqrt_P[i] # sigma point scaling
            sigma_points[self.n + i + 1] = self.x - self.gamma * sqrt_P[i]
        return sigma_points

    def predict(self):
        sigma_points = self.generate_sigma_points()
        predicted_sigma = np.array([self.process_model(s) for s in sigma_points]) 
        self.x = np.sum(self.weights_mean[:, None] * predicted_sigma, axis=0)
        self.P = np.sum(
            [self.weights_cov[i] * np.outer(predicted_sigma[i] - self.x, predicted_sigma[i] - self.x) for i in range(2 * self.n + 1)], axis=0
        ) + self.Q

    def update(self, z):
        sigma_points = self.generate_sigma_points() 
        predicted_measurements = np.array([self.measurement_model(s) for s in sigma_points])
        z_hat = np.sum(self.weights_mean[:, None] * predicted_measurements, axis=0) # measurement mean
        S = np.sum(
            [self.weights_cov[i] * np.outer(predicted_measurements[i] - z_hat, predicted_measurements[i] - z_hat) for i in range(2 * self.n + 1)], axis=0
        ) + self.R # measurement covariance
        
        cross_covariance = np.sum(
            [self.weights_cov[i] * np.outer(sigma_points[i] - self.x, predicted_measurements[i] - z_hat) for i in range(2 * self.n + 1)], axis=0
        )
        K = cross_covariance @ np.linalg.inv(S) # kalman gain
        self.x += K @ (z - z_hat) # state update
        self.P -= K @ S @ K.T # covariance update

    def process_model(self, state):
        x, y, theta, v, w, vdot = state
        dt = self.dt

        new_theta = theta + w * dt
        new_x = x + v * np.cos(theta) * dt # update x
        new_y = y + v * np.sin(theta) * dt # update y
        new_v = v + vdot * dt # update velocity

        return np.array([new_x, new_y, new_theta, new_v, w, vdot])

    def measurement_model(self, state):
        x, y, theta, v, w, vdot = state

        ax = vdot
        ay = v * w

        return np.array([v, w, ax, ay])

    def get_states(self):
        return self.x
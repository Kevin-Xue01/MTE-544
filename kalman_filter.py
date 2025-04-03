import numpy as np

class kalman_filter:
       
    def __init__(self, P,Q,R, x, dt):
        self.P=P
        self.Q=Q
        self.R=R
        self.x=x
        self.dt=dt
     
    def predict(self):

        self.A = self.jacobian_A()  # Computed with previous state

        self.motion_model()         # Predict state

        self.C = self.jacobian_H()  # Compuuted with predicted state
        
        self.P = self.A @ self.P @ self.A.T + self.Q # Predicted estimation covariance

    def update(self, z):
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)  # Kalman gain
        
        self.x = self.x + K @ (z - self.measurement_model())                        # Updated state estimate

        self.P = (np.eye(self.A.shape[0]) - K @ self.C) @ self.P                    # Updated estimation covariance

    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        ax = vdot
        ay = v * w

        return np.array([v,w,ax,ay])
        
    def motion_model(self):
        
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            th + w * dt,
            w,
            v  + vdot*dt,
            vdot,
        ])
    
    def jacobian_A(self):
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        return np.array([
            #x, y,                  th, w,             v, vdot
            [1, 0,-v * np.sin(th) * dt, 0, np.cos(th)*dt,  0],
            [0, 1, v * np.cos(th) * dt, 0, np.sin(th)*dt,  0],
            [0, 0,                1, dt,           0,  0],
            [0, 0,                0, 1,            0,  0],
            [0, 0,                0, 0,            1,  dt],
            [0, 0,                0, 0,            0,  1 ]
        ])
    
     
    def jacobian_H(self):
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x, y,th, w, v,vdot
            [0,0,0  , 0, 1, 0], # v
            [0,0,0  , 1, 0, 0], # w
            [0,0,0  , 0, 0, 1], # ax
            [0,0,0  , v, w, 0], # ay
        ])
         
    def get_states(self):
        return self.x

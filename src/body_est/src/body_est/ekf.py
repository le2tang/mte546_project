import numpy as np


class EKF:
    def _init_(self, model):
        self.model = model

        self.state = np.zeros((13, 1))  # [p, v, q, w]
        self.state_covar = np.eye(13)
        self.process_noise = np.eye(13)
        self.measurement_noise = np.eye(7) # does the number of states = number of measurements?

    def predict(self):
        """Update prior state estimate and covariance"""
        A = self.model.forward_jacobian(self.state)
        
        self.state = self.model.forward_model(self.state)
        
        self.state_covar = A * self.state_covar * A.T + self.process_noise

        return self.state, self.state_covar

    def correct(self, measurement):
        """Update prior state estimate and covariance"""
        self.model_meas = self.model.measurement_model(self.state)
        
        measurement_residual = measurement - self.model_meas
        
        # Calculate the measurement residual covariance
        C = self.model.measurement_jacobian(self.state)
        S_k = C * self.state_covar * C.T + self.measurement_noise
        
        # Calculate the Kalman gain
        K_k = self.state_covar * C.T * np.linalg.pinv(S_k)
        
        # Update state estimate for time k
        self.state = self.state + K_k * measurement_residual
        
        # Update state covariance estimate for time k
        self.state_covar = self.state_covar - K_k * C * self.state_covar

        return self.state, self.state_covar

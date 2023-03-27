import numpy as np


class EKF:
    def __init__(self, model):
        self.model = model

        self.state = self.model.init_state()
        self.state_covar = self.model.init_state_covar()
        self.process_noise = self.model.process_noise()
        self.measurement_noise = self.model.measurement_noise()

    def predict(self):
        """Update prior state estimate and covariance"""
        A = self.model.forward_jacobian(self.state)

        self.state = self.model.forward_model(self.state)

        self.state_covar = A @ self.state_covar @ A.T + self.process_noise

        return self.state, self.state_covar

    def correct(self, measurement):
        """Update prior state estimate and covariance"""
        self.model_meas = self.model.measurement_model(self.state)

        measurement_residual = (
            measurement.reshape(self.model_meas.shape) - self.model_meas
        )

        # Calculate the measurement residual covariance
        C = self.model.measurement_jacobian(self.state)
        S_k = C @ self.state_covar @ C.T + self.measurement_noise

        # Calculate the Kalman gain
        K_k = self.state_covar @ C.T @ np.linalg.pinv(S_k)

        # Update state estimate for time k
        self.state = self.state + K_k @ measurement_residual

        # Update state covariance estimate for time k
        self.state_covar = self.state_covar - K_k @ C @ self.state_covar

        return self.state, self.state_covar

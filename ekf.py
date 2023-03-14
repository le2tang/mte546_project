import numpy as np


class EKF:
    def _init_(self, model):
        self.model = model

        self.state = np.zeros((13, 1))  # [p, v, q, w]
        self.state_covar = np.eye(13)

    def predict(self):
        """Update prior state estimate and covariance"""
        self.state = self.model.forward_model(self.state)

        # Update state covariance
        raise NotImplementedError()

    def correct(self, measurement):
        """Update prior state estimate and covariance"""
        self.model_meas = self.model.measurement_model(self.state)

        # Update state covariance from kalman gain
        raise NotImplementedError()
